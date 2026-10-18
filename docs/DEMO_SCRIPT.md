# RAG Based Profile Matching: demo video plan (3 to 4 minutes)

Have a terminal and VS Code open in the project folder, with `vector_db/` already
built so nothing has to download during the recording. Increase the terminal font
size before recording.

## 0:00 – 0:30  The problem and the plan

Show the README diagram.

> "Recruiters get hundreds of resumes per job. This project reads resumes in PDF,
> Word and text format, stores them in a vector database, and for a job description
> returns the ten best candidates with a score and a reason. It has two parts:
> ingestion and matching."

## 0:30 – 1:15  Part A: ingestion

Show the `data/resumes` folder (mixed file types), then open one resume.

```bash
python3 resume_rag.py
```

> "Each resume is read with the file tools from Milestone 1 and split at its section
> headings, so one chunk is one section like experience or skills. Headings are
> written differently in different resumes, so there is a lookup table for them.
> For each resume I pull out the name, skills, years of experience and education,
> and store them as metadata next to every chunk in ChromaDB."

Point at the example record in the output.

## 1:15 – 2:30  Part B: matching

Open `data/job_descriptions/jd_01_python_backend.txt`, then run:

```bash
python3 job_matcher.py data/job_descriptions/jd_01_python_backend.txt
```

> "The job asks for Python and Django and at least five years. First the filter drops
> everyone who misses a must-have. Then two searches run: semantic search on the
> embeddings, and keyword search with BM25 where must-have skills count double. The
> two rankings are merged with Reciprocal Rank Fusion, and the top ten get a score out
> of 100: meaning, skills and experience."

Open `output/matches_jd_01_python_backend.json` and walk through one candidate:
score, matched skills, excerpts with their section names, reasoning.

Then show the filter's effect:

```bash
python3 job_matcher.py data/job_descriptions/jd_01_python_backend.txt --no-filter
```

> "Without the filter, people with only one year or without Django still show up,
> because their resumes read almost the same. Hard requirements are rules, so the
> system treats them as rules."

Optional, if time allows:

```bash
python3 job_matcher.py --text "DevOps engineer, 4+ years K8s, Terraform on AWS"
```

> "Free text works too, and K8s is understood as Kubernetes."

## 2:30 – 3:30  Results

```bash
python3 evaluate.py
```

Open `output/evaluation_report.md` or scroll to section 4 of the notebook.

> "Since the dataset is generated, I know the right answers. Metadata extraction is
> checked field by field. To compare the search methods fairly I switch the filter
> off, otherwise every method looks perfect. Here are Precision at 5, Recall at 10 and
> MRR for semantic, keyword and hybrid. Search takes about [read the number] milliseconds."

Say one real observation from your own numbers, for example which method had the
best MRR.

## 3:30 – 4:00  Limits and wrap-up

> "The main limits: the dataset is cleaner than real resumes, skills come only from
> the skills section and the skill list is fixed. Next steps would be a cross-encoder
> re-ranker for bigger pools and reading skills from experience lines too. Thanks for
> watching."
