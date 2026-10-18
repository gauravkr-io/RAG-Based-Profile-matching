"""RAG Based Profile Matching, Part B: job matcher (retrieval and scoring).

Give it a job description and it returns the 10 best matching candidates from
the resume database built by resume_rag.py.

How one search works:
  1. Read the job description: title, must-have skills, nice-to-have skills,
     minimum years of experience
  2. Filter: drop every candidate who lacks a must-have skill or has too few years
  3. Semantic search: embed the job description and ask ChromaDB for the
     closest resume sections (only from candidates who passed the filter)
  4. Keyword search: BM25 over skills and resume words, with must-have skills
     counted twice
  5. Combine both rankings with Reciprocal Rank Fusion (RRF) and keep the top 10
  6. Give each of the 10 a score from 0 to 100, short excerpts and a reason
  7. Save everything as JSON in output/

Examples:
    python job_matcher.py data/job_descriptions/jd_01_python_backend.txt
    python job_matcher.py --all
    python job_matcher.py --text "Need a backend developer with 5+ years Python and Django"
    python job_matcher.py data/job_descriptions/jd_02_ml_engineer.txt --top-k 5 --llm
"""
import argparse
import json
import os
import re
import time

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from resume_rag import (COLLECTION_NAME, DB_FOLDER, EMBEDDING_MODEL, KNOWN_SKILLS,
                        find_skills, skill_key)

TOP_K = 10
RRF_K = 60                    # standard RRF constant. Bigger values flatten the rank differences.
SECTION_MATCH_SIMILARITY = 0.30   # a section this close to the job counts as "matched"

# How the 0-100 score is split
SEMANTIC_POINTS = 45          # how close the resume text is in meaning
SKILL_POINTS = 35             # how many of the job's skills the candidate lists
EXPERIENCE_POINTS = 20        # years compared with the minimum asked for

# Cosine similarity between a job description and a resume section usually lands
# between about 0.2 (unrelated) and 0.65 (very close) with this model.
# These two numbers stretch that range to 0..1 for the score.
SIMILARITY_LOW = 0.20
SIMILARITY_HIGH = 0.65


# ---------------------------------------------------------------------------
# Loading the database (done once, then reused)
# ---------------------------------------------------------------------------
_loaded = {}


def load_resources():
    """Open ChromaDB, read candidates.json and load the embedding model, only the first time."""
    if _loaded:
        return _loaded

    candidates_file = os.path.join(DB_FOLDER, "candidates.json")
    if not os.path.exists(candidates_file):
        raise SystemExit("No resume database found. Run: python resume_rag.py")

    with open(candidates_file, "r", encoding="utf-8") as file:
        candidates = json.load(file)

    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # The BM25 index is built over ALL candidates, so word rarity is measured
    # across the whole pool, not only the filtered group.
    tokenized = []
    for candidate in candidates:
        tokenized.append(keyword_tokens(candidate["full_text"]))
    bm25 = BM25Okapi(tokenized)

    _loaded["candidates"] = candidates
    _loaded["collection"] = collection
    _loaded["model"] = model
    _loaded["bm25"] = bm25
    return _loaded


def keyword_tokens(text):
    """Words for BM25, plus one special token per known skill.

    The special token (for example "skill_kubernetes") is the same whether the
    resume wrote "Kubernetes" or "K8s", so BM25 matches both spellings.
    """
    words = re.findall(r"[a-z0-9+#]+", text.lower())
    for skill in find_skills(text):
        words.append(skill_key(skill).replace("has_", "skill_"))
    return words


# ---------------------------------------------------------------------------
# Step 1: read the job description
# ---------------------------------------------------------------------------
def split_skill_list(line_value):
    """'Python, Django and AWS' -> standard skill names found in it."""
    return find_skills(line_value)


def parse_job_description(text):
    """Pull out title, must-have skills, nice-to-have skills and minimum years.

    Our job description files use labelled lines ("Must have: ..."). A pasted
    free-text job description has no labels, so then we look for patterns like
    "5+ years Python" and treat every other skill mentioned as nice to have.
    """
    job = {"title": "", "must_have": [], "nice_to_have": [], "min_years": 0, "text": text.strip()}

    labelled = False
    for line in text.split("\n"):
        lower = line.lower().strip()
        value = line.split(":", 1)[1] if ":" in line else ""
        if lower.startswith("title:"):
            job["title"] = value.strip()
        elif lower.startswith("must have:") or lower.startswith("required:"):
            job["must_have"] = split_skill_list(value)
            labelled = True
        elif lower.startswith("nice to have:") or lower.startswith("preferred:"):
            job["nice_to_have"] = split_skill_list(value)
        elif lower.startswith("minimum experience:"):
            number = re.search(r"\d+", value)
            if number:
                job["min_years"] = int(number.group())

    if not labelled:
        # Free text. "5+ years Python" or "5 years of experience in Python"
        # means Python is a must-have and 5 is the minimum.
        pattern = r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience\s+)?(?:in\s+|with\s+)?([A-Za-z][A-Za-z0-9.+#/ ]{1,30})"
        for years, after in re.findall(pattern, text, re.IGNORECASE):
            skills_after = find_skills(after)
            if skills_after:
                job["must_have"].append(skills_after[0])
            job["min_years"] = max(job["min_years"], int(years))
        # A plain "Minimum 4 years" without a skill still sets the minimum
        if job["min_years"] == 0:
            number = re.search(r"(\d+)\+?\s*years", text, re.IGNORECASE)
            if number:
                job["min_years"] = int(number.group(1))
        for skill in find_skills(text):
            if skill not in job["must_have"]:
                job["nice_to_have"].append(skill)

    if job["title"] == "":
        job["title"] = text.strip().split("\n")[0][:60]

    # A skill listed under both is only kept as a must-have
    nice = []
    for skill in job["nice_to_have"]:
        if skill not in job["must_have"]:
            nice.append(skill)
    job["nice_to_have"] = nice
    return job


# ---------------------------------------------------------------------------
# Step 2: must-have filter
# ---------------------------------------------------------------------------
def passes_filter(candidate, job):
    for skill in job["must_have"]:
        if skill not in candidate["skills"]:
            return False
    return candidate["experience_years"] >= job["min_years"]


def chroma_filter(job):
    """The same filter written in ChromaDB's language, so the vector search
    only returns sections from candidates who qualify."""
    conditions = []
    for skill in job["must_have"]:
        conditions.append({skill_key(skill): True})
    if job["min_years"] > 0:
        conditions.append({"experience_years": {"$gte": job["min_years"]}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Steps 3 to 5: semantic search, keyword search, RRF
# ---------------------------------------------------------------------------
def semantic_search(job, use_filter):
    """Return {resume_path: list of (similarity, section, chunk_text)} best first."""
    resources = load_resources()
    collection = resources["collection"]

    query_text = job["title"] + "\n" + job["text"]
    query_vector = resources["model"].encode(query_text, normalize_embeddings=True)

    where = chroma_filter(job) if use_filter else None
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=collection.count(),      # small database, so ask for every section
        where=where,
    )

    # ChromaDB returns lists inside a list (one inner list per query). We sent one query.
    ids = result["ids"][0]
    distances = result["distances"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    sections_by_candidate = {}
    for position in range(len(ids)):
        # Cosine distance = 1 - cosine similarity, so we turn it back
        similarity = 1 - distances[position]
        path = metadatas[position]["resume_path"]
        if path not in sections_by_candidate:
            sections_by_candidate[path] = []
        sections_by_candidate[path].append((similarity, metadatas[position]["section"], documents[position]))

    for path in sections_by_candidate:
        sections_by_candidate[path].sort(reverse=True)    # highest similarity first
    return sections_by_candidate


def keyword_scores(job):
    """BM25 score for every candidate. Must-have skills appear twice in the query."""
    resources = load_resources()
    query = []
    for skill in job["must_have"]:
        token = skill_key(skill).replace("has_", "skill_")
        query += [token, token]
    for skill in job["nice_to_have"]:
        query.append(skill_key(skill).replace("has_", "skill_"))
    query += re.findall(r"[a-z0-9+#]+", job["title"].lower())

    scores = resources["bm25"].get_scores(query)
    by_path = {}
    for number in range(len(resources["candidates"])):
        by_path[resources["candidates"][number]["resume_path"]] = float(scores[number])
    return by_path


def rrf_combine(rankings):
    """Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank) over every ranking."""
    rrf = {}
    for ranking in rankings:
        for position in range(len(ranking)):
            path = ranking[position]
            rank = position + 1
            rrf[path] = rrf.get(path, 0) + 1 / (RRF_K + rank)
    return sorted(rrf, key=rrf.get, reverse=True)


def rank_candidates(job, mode="hybrid", use_filter=True):
    """Return (ranked list of resume paths, semantic sections, BM25 scores).

    mode is "hybrid" (default), "semantic" or "keyword". The last two exist so
    evaluate.py can compare them.
    """
    resources = load_resources()

    allowed = []
    for candidate in resources["candidates"]:
        if not use_filter or passes_filter(candidate, job):
            allowed.append(candidate["resume_path"])

    sections = semantic_search(job, use_filter)
    semantic_ranking = sorted(
        [path for path in sections if path in allowed],
        key=lambda path: sections[path][0][0],      # the candidate's best section similarity
        reverse=True,
    )

    bm25 = keyword_scores(job)
    keyword_ranking = []
    for path in sorted(bm25, key=bm25.get, reverse=True):
        # A score of 0 means no word in common, so it earns no place in the keyword list
        if path in allowed and bm25[path] > 0:
            keyword_ranking.append(path)

    if mode == "semantic":
        ranked = semantic_ranking
    elif mode == "keyword":
        ranked = keyword_ranking
    else:
        ranked = rrf_combine([semantic_ranking, keyword_ranking])
    return ranked, sections, bm25


# ---------------------------------------------------------------------------
# Step 6: score, excerpts and reasoning
# ---------------------------------------------------------------------------
def score_candidate(candidate, job, candidate_sections):
    best_similarity = candidate_sections[0][0] if candidate_sections else 0.0
    semantic_part = (best_similarity - SIMILARITY_LOW) / (SIMILARITY_HIGH - SIMILARITY_LOW)
    semantic_part = min(max(semantic_part, 0.0), 1.0)          # keep it between 0 and 1

    # Must-have skills count double. With the filter on, they are always present.
    matched_must = [skill for skill in job["must_have"] if skill in candidate["skills"]]
    matched_nice = [skill for skill in job["nice_to_have"] if skill in candidate["skills"]]
    possible = 2 * len(job["must_have"]) + len(job["nice_to_have"])
    if possible == 0:
        skill_part = 1.0
    else:
        skill_part = (2 * len(matched_must) + len(matched_nice)) / possible

    if job["min_years"] == 0:
        experience_part = 1.0
    else:
        # Meeting the minimum gives two thirds of the points. 1.5 times the minimum gives all of them.
        experience_part = min(candidate["experience_years"] / job["min_years"], 1.5) / 1.5

    score = (SEMANTIC_POINTS * semantic_part
             + SKILL_POINTS * skill_part
             + EXPERIENCE_POINTS * experience_part)

    return {
        "score": round(score, 1),
        "best_similarity": round(best_similarity, 3),
        "matched_must": matched_must,
        "matched_nice": matched_nice,
    }


def matched_sections(candidate_sections, job):
    """Sections that are close in meaning, or that mention one of the job's skills."""
    names = []
    for similarity, section, text in candidate_sections:
        mentions_skill = False
        for skill in find_skills(text):
            if skill in job["must_have"] or skill in job["nice_to_have"]:
                mentions_skill = True
        if (similarity >= SECTION_MATCH_SIMILARITY or mentions_skill) and section not in names:
            names.append(section)
    if not names and candidate_sections:
        names.append(candidate_sections[0][1])
    return names


def excerpts(candidate_sections, count=2):
    """The text of the closest sections, without the 'Name | section' header line."""
    result = []
    for similarity, section, text in candidate_sections[:count]:
        body = text.split("\n", 1)[1] if "\n" in text else text
        body = " ".join(body.split())          # put it on one line
        if len(body) > 300:
            body = body[:297] + "..."
        result.append("[" + section + "] " + body)
    return result


def rule_based_reasoning(candidate, job, details, sections_matched):
    parts = []
    if job["min_years"] > 0:
        parts.append(f"{candidate['experience_years']} years of experience against the {job['min_years']}+ asked for.")
    else:
        parts.append(f"{candidate['experience_years']} years of experience.")

    if job["must_have"]:
        missing_must = [s for s in job["must_have"] if s not in details["matched_must"]]
        if missing_must:
            parts.append("Missing must-have skills: " + ", ".join(missing_must) + ".")
        else:
            parts.append("Has every must-have skill (" + ", ".join(job["must_have"]) + ").")

    if job["nice_to_have"]:
        missing_nice = [s for s in job["nice_to_have"] if s not in details["matched_nice"]]
        sentence = f"Covers {len(details['matched_nice'])} of {len(job['nice_to_have'])} nice-to-have skills"
        if details["matched_nice"]:
            sentence += " (" + ", ".join(details["matched_nice"]) + ")"
        if missing_nice:
            sentence += ", without " + ", ".join(missing_nice)
        parts.append(sentence + ".")

    parts.append("Sections that matched the job: " + ", ".join(sections_matched)
                 + f". Closest section similarity {details['best_similarity']}.")
    return " ".join(parts)


def llm_reasoning(candidate, job, details, sections_matched, excerpt_list):
    """Ask an LLM (through OpenRouter) for a two-sentence recruiter note.

    Only used with --llm. If anything goes wrong, the rule-based reason is used instead.
    """
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        if not os.getenv("OPENROUTER_API_KEY"):
            return None
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        facts = {
            "job_title": job["title"],
            "must_have": job["must_have"],
            "nice_to_have": job["nice_to_have"],
            "min_years": job["min_years"],
            "candidate": candidate["candidate_name"],
            "experience_years": candidate["experience_years"],
            "candidate_skills": candidate["skills"],
            "sections_matched": sections_matched,
            "excerpts": excerpt_list,
            "match_score": details["score"],
        }
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You write short notes for recruiters. Use only the facts given. "
                                              "Two sentences: why this candidate fits, then the main gap if there is one. "
                                              "Mention which resume sections matched."},
                {"role": "user", "content": json.dumps(facts)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as error:
        print("  (LLM reasoning failed, using the rule-based reason:", str(error)[:80], ")")
        return None


# ---------------------------------------------------------------------------
# The full search
# ---------------------------------------------------------------------------
def match_job(job_text, top_k=TOP_K, mode="hybrid", use_filter=True, use_llm=False):
    """Run one search. Returns (result in the required JSON format, details for analysis)."""
    start = time.time()
    resources = load_resources()
    job = parse_job_description(job_text)

    ranked, sections, bm25 = rank_candidates(job, mode=mode, use_filter=use_filter)
    shortlist = ranked[:top_k]

    candidates_by_path = {}
    for candidate in resources["candidates"]:
        candidates_by_path[candidate["resume_path"]] = candidate

    matches = []
    for path in shortlist:
        candidate = candidates_by_path[path]
        candidate_sections = sections.get(path, [])
        details = score_candidate(candidate, job, candidate_sections)
        section_names = matched_sections(candidate_sections, job)
        excerpt_list = excerpts(candidate_sections)

        reasoning = None
        if use_llm:
            reasoning = llm_reasoning(candidate, job, details, section_names, excerpt_list)
        if not reasoning:
            reasoning = rule_based_reasoning(candidate, job, details, section_names)

        matches.append({
            "candidate_name": candidate["candidate_name"],
            "resume_path": path,
            "match_score": int(round(details["score"])),
            "matched_skills": details["matched_must"] + details["matched_nice"],
            "relevant_excerpts": excerpt_list,
            "reasoning": reasoning,
        })

    # RRF picked the shortlist. The final list is shown in score order, so the
    # numbers a recruiter sees always go from high to low.
    matches.sort(key=lambda match: match["match_score"], reverse=True)

    seconds = time.time() - start
    output = {"job_description": job["text"], "top_matches": matches}
    info = {"job": job, "ranked_paths": ranked, "seconds": seconds}
    return output, info


def print_summary(output, info):
    job = info["job"]
    print("\nJob:", job["title"])
    print("  Must have:   ", ", ".join(job["must_have"]) or "(none)")
    print("  Nice to have:", ", ".join(job["nice_to_have"]) or "(none)")
    print("  Min years:   ", job["min_years"])
    print("  Search took  ", round(info["seconds"] * 1000), "ms")
    if not output["top_matches"]:
        print("  No candidate passed the must-have filter.")
    for number, match in enumerate(output["top_matches"], start=1):
        print(f"  {number:2}. {match['candidate_name']:<22} score {match['match_score']:3}  "
              f"skills: {', '.join(match['matched_skills'])}")


def save_output(output, name):
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "matches_" + name + ".json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print("  Saved", path)


def main():
    parser = argparse.ArgumentParser(description="Find the best candidates for a job description.")
    parser.add_argument("jd_file", nargs="?", help="a job description text file")
    parser.add_argument("--text", help="paste a job description directly instead of a file")
    parser.add_argument("--all", action="store_true", help="run every file in data/job_descriptions")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="how many candidates to return (default 10)")
    parser.add_argument("--no-filter", action="store_true", help="do not drop candidates missing must-haves")
    parser.add_argument("--llm", action="store_true", help="write the reasoning with an LLM through OpenRouter")
    args = parser.parse_args()

    jobs = []     # (name for the output file, job description text)
    if args.all:
        folder = "data/job_descriptions"
        for file_name in sorted(os.listdir(folder)):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder, file_name), "r", encoding="utf-8") as file:
                    jobs.append((os.path.splitext(file_name)[0], file.read()))
    elif args.text:
        jobs.append(("custom", args.text))
    elif args.jd_file:
        with open(args.jd_file, "r", encoding="utf-8") as file:
            jobs.append((os.path.splitext(os.path.basename(args.jd_file))[0], file.read()))
    else:
        parser.print_help()
        return

    for name, text in jobs:
        output, info = match_job(text, top_k=args.top_k, use_filter=not args.no_filter, use_llm=args.llm)
        print_summary(output, info)
        save_output(output, name)


if __name__ == "__main__":
    main()
