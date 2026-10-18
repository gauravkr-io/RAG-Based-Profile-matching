"""Measure how well the matcher works, using data/ground_truth.json.

Because the dataset is generated, we know the right answers:
  qualified  candidates who have every must-have skill and enough years
  strong     qualified, same role family as the job, and at least 2 nice-to-have skills
  family     the kind of role each candidate does (python_backend, devops, ...)

What gets measured:
  1. Metadata extraction: did we read name, skills, years and education correctly?
  2. Must-have filter: does it keep exactly the qualified candidates?
  3. Retrieval quality WITHOUT the filter, for semantic, keyword and hybrid search.
     (With the filter on, all three would look perfect, because the filter does
     the hard work. Turning it off shows what each search method finds by itself.)
       Precision@5   share of the top 5 that are in the right role family
       Recall@10     share of the qualified candidates that appear in the top 10
       MRR           1 / rank of the first strong match (1.0 means it was ranked first)
  4. Full pipeline: is the candidate ranked first a strong match?
  5. Speed: ingestion time and search latency

Run after resume_rag.py:
    python evaluate.py
"""
import json
import os
import time

import numpy as np

from resume_rag import CURRENT_YEAR
from job_matcher import load_resources, match_job, parse_job_description, passes_filter, rank_candidates

JD_FOLDER = "data/job_descriptions"
LATENCY_RUNS = 5          # each job description is searched this many times to measure speed


def load_ground_truth():
    with open("data/ground_truth.json", "r", encoding="utf-8") as file:
        truth = json.load(file)
    # Years of experience are counted up to the current year. If the dataset was made
    # in an earlier year, every "present" job is now one year longer than the truth says.
    if truth.get("generated_in_year", CURRENT_YEAR) != CURRENT_YEAR:
        raise SystemExit("The dataset was generated in " + str(truth["generated_in_year"]) +
                         ". Run python generate_dataset.py and python resume_rag.py again.")
    return truth


def load_jobs():
    jobs = []
    for file_name in sorted(os.listdir(JD_FOLDER)):
        if file_name.endswith(".txt"):
            with open(os.path.join(JD_FOLDER, file_name), "r", encoding="utf-8") as file:
                jobs.append((os.path.splitext(file_name)[0], file.read()))
    return jobs


# The role family each job belongs to, taken from the job id, e.g. jd_04_devops -> devops
def family_of_job(job_id):
    return job_id.split("_", 2)[2]


def extraction_accuracy(truth):
    candidates = load_resources()["candidates"]
    truth_by_path = {}
    for record in truth["candidates"]:
        truth_by_path[record["resume_path"]] = record

    correct = {"name": 0, "skills": 0, "experience_years": 0, "education": 0}
    for candidate in candidates:
        real = truth_by_path[candidate["resume_path"]]
        if candidate["candidate_name"] == real["name"]:
            correct["name"] += 1
        if set(candidate["skills"]) == set(real["skills"]):
            correct["skills"] += 1
        if candidate["experience_years"] == real["experience_years"]:
            correct["experience_years"] += 1
        if candidate["education"] == real["education"]:
            correct["education"] += 1

    accuracy = {}
    for field in correct:
        accuracy[field] = round(correct[field] / len(candidates), 3)
    return accuracy


def filter_quality(jobs, truth):
    candidates = load_resources()["candidates"]
    rows = []
    for job_id, text in jobs:
        job = parse_job_description(text)
        kept = set()
        for candidate in candidates:
            if passes_filter(candidate, job):
                kept.add(candidate["resume_path"])
        qualified = set(truth["jobs"][job_id]["qualified"])
        correct = len(kept & qualified)
        rows.append({
            "job": job_id,
            "kept": len(kept),
            "qualified": len(qualified),
            "precision": round(correct / len(kept), 3) if kept else 0.0,
            "recall": round(correct / len(qualified), 3) if qualified else 1.0,
        })
    return rows


def retrieval_quality(jobs, truth):
    family_by_path = {}
    for record in truth["candidates"]:
        family_by_path[record["resume_path"]] = record["family"]

    results = {}
    for mode in ["semantic", "keyword", "hybrid"]:
        per_job = []
        for job_id, text in jobs:
            job = parse_job_description(text)
            ranked, sections, bm25 = rank_candidates(job, mode=mode, use_filter=False)
            qualified = set(truth["jobs"][job_id]["qualified"])
            strong = set(truth["jobs"][job_id]["strong"])
            family = family_of_job(job_id)

            top5 = ranked[:5]
            same_family = [path for path in top5 if family_by_path[path] == family]
            precision_at_5 = len(same_family) / 5

            top10 = set(ranked[:10])
            recall_at_10 = len(top10 & qualified) / len(qualified) if qualified else 1.0

            reciprocal_rank = 0.0
            for position in range(len(ranked)):
                if ranked[position] in strong:
                    reciprocal_rank = 1 / (position + 1)
                    break

            per_job.append({
                "job": job_id,
                "precision_at_5": round(precision_at_5, 3),
                "recall_at_10": round(recall_at_10, 3),
                "mrr": round(reciprocal_rank, 3),
            })

        results[mode] = {
            "per_job": per_job,
            "precision_at_5": round(float(np.mean([row["precision_at_5"] for row in per_job])), 3),
            "recall_at_10": round(float(np.mean([row["recall_at_10"] for row in per_job])), 3),
            "mrr": round(float(np.mean([row["mrr"] for row in per_job])), 3),
        }
    return results


def full_pipeline_quality(jobs, truth):
    rows = []
    for job_id, text in jobs:
        output, info = match_job(text)
        strong = set(truth["jobs"][job_id]["strong"])
        returned = [match["resume_path"] for match in output["top_matches"]]
        top_is_strong = bool(returned) and returned[0] in strong
        rows.append({
            "job": job_id,
            "returned": len(returned),
            "top_candidate": output["top_matches"][0]["candidate_name"] if returned else None,
            "top_score": output["top_matches"][0]["match_score"] if returned else None,
            "top_is_strong_match": top_is_strong,
        })
    hit_rate = sum(1 for row in rows if row["top_is_strong_match"]) / len(rows)
    return rows, round(hit_rate, 3)


def latency(jobs):
    match_job(jobs[0][1])          # warm-up run, so model loading is not counted
    timings = []
    for job_id, text in jobs:
        for _ in range(LATENCY_RUNS):
            start = time.time()
            match_job(text)
            timings.append((time.time() - start) * 1000)
    return {
        "searches": len(timings),
        "mean_ms": round(float(np.mean(timings)), 1),
        "p95_ms": round(float(np.percentile(timings, 95)), 1),
        "max_ms": round(float(np.max(timings)), 1),
    }


def write_report(metrics):
    lines = ["# Evaluation report", ""]
    lines.append(f"Dataset: {metrics['dataset']['resumes']} resumes, {metrics['dataset']['chunks']} section chunks, "
                 f"{metrics['dataset']['job_descriptions']} job descriptions.")
    lines += ["", "## 1. Metadata extraction accuracy", "", "| Field | Accuracy |", "|---|---|"]
    for field, value in metrics["extraction_accuracy"].items():
        lines.append(f"| {field} | {value:.0%} |")

    lines += ["", "## 2. Must-have filter", "", "| Job | Kept | Qualified | Precision | Recall |", "|---|---|---|---|---|"]
    for row in metrics["filter"]:
        lines.append(f"| {row['job']} | {row['kept']} | {row['qualified']} | {row['precision']:.2f} | {row['recall']:.2f} |")

    lines += ["", "## 3. Retrieval quality without the filter", "",
              "| Method | Precision@5 | Recall@10 | MRR |", "|---|---|---|---|"]
    for mode in ["semantic", "keyword", "hybrid"]:
        result = metrics["retrieval"][mode]
        lines.append(f"| {mode} | {result['precision_at_5']:.2f} | {result['recall_at_10']:.2f} | {result['mrr']:.2f} |")

    lines += ["", "Per job, hybrid:", "", "| Job | Precision@5 | Recall@10 | MRR |", "|---|---|---|---|"]
    for row in metrics["retrieval"]["hybrid"]["per_job"]:
        lines.append(f"| {row['job']} | {row['precision_at_5']:.2f} | {row['recall_at_10']:.2f} | {row['mrr']:.2f} |")

    lines += ["", "## 4. Full pipeline (filter + hybrid + score)", "",
              f"Top candidate is a strong match for {metrics['top1_strong_rate']:.0%} of the job descriptions.", "",
              "| Job | Returned | Top candidate | Score | Strong match |", "|---|---|---|---|---|"]
    for row in metrics["pipeline"]:
        lines.append(f"| {row['job']} | {row['returned']} | {row['top_candidate']} | {row['top_score']} | "
                     f"{'yes' if row['top_is_strong_match'] else 'no'} |")

    speed = metrics["latency"]
    ingest = metrics["ingestion"]
    lines += ["", "## 5. Speed", "",
              f"- Ingestion of all resumes: {ingest.get('total_ingestion_seconds', 'n/a')} s "
              f"(embedding part {ingest.get('embedding_seconds', 'n/a')} s)",
              f"- Search latency over {speed['searches']} searches: mean {speed['mean_ms']} ms, "
              f"p95 {speed['p95_ms']} ms, max {speed['max_ms']} ms", ""]

    with open("output/evaluation_report.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    truth = load_ground_truth()
    jobs = load_jobs()
    resources = load_resources()

    ingestion = {}
    if os.path.exists("output/ingestion_stats.json"):
        with open("output/ingestion_stats.json", "r", encoding="utf-8") as file:
            ingestion = json.load(file)

    print("Checking metadata extraction...")
    extraction = extraction_accuracy(truth)
    print("Checking the must-have filter...")
    filter_rows = filter_quality(jobs, truth)
    print("Comparing semantic, keyword and hybrid search...")
    retrieval = retrieval_quality(jobs, truth)
    print("Running the full pipeline...")
    pipeline_rows, top1_rate = full_pipeline_quality(jobs, truth)
    print("Measuring search speed...")
    speed = latency(jobs)

    metrics = {
        "dataset": {
            "resumes": len(resources["candidates"]),
            "chunks": resources["collection"].count(),
            "job_descriptions": len(jobs),
        },
        "extraction_accuracy": extraction,
        "filter": filter_rows,
        "retrieval": retrieval,
        "pipeline": pipeline_rows,
        "top1_strong_rate": top1_rate,
        "latency": speed,
        "ingestion": ingestion,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    write_report(metrics)

    print("\nExtraction accuracy:", extraction)
    print("\nRetrieval without filter      P@5    R@10   MRR")
    for mode in ["semantic", "keyword", "hybrid"]:
        result = retrieval[mode]
        print(f"  {mode:<26}  {result['precision_at_5']:.2f}   {result['recall_at_10']:.2f}   {result['mrr']:.2f}")
    print("\nTop candidate is a strong match:", f"{top1_rate:.0%}", "of jobs")
    print("Search latency: mean", speed["mean_ms"], "ms, p95", speed["p95_ms"], "ms")
    print("\nSaved output/metrics.json and output/evaluation_report.md")


if __name__ == "__main__":
    main()
