# RAG Based Profile Matching: setup and running

Tested with Python 3.10 to 3.13 on macOS and Linux. On Windows, use `python`
instead of `python3` and `.venv\Scripts\activate` to activate the environment.

## 1. Create a virtual environment and install

From inside the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The install takes a few minutes the first time, mostly for `sentence-transformers`
(it brings PyTorch with it).

## 2. Optional: API key for LLM reasoning

Only needed if you want to try `--llm`. Everything else works without a key.

```bash
cp .env.example .env
```

Then open `.env` and put your OpenRouter key after `OPENROUTER_API_KEY=`.

## 3. Generate the dataset

The `data/` folder already contains the generated files. To recreate them:

```bash
python3 generate_dataset.py
```

Expected output:

```
Created 36 resumes in data/resumes
Created 6 job descriptions in data/job_descriptions
  jd_01_python_backend: 3 qualified, 3 strong matches
  jd_02_ml_engineer: 4 qualified, 2 strong matches
  ...
```

## 4. Build the resume database (Part A)

```bash
python3 resume_rag.py
```

The first run downloads the embedding model (about 90 MB), so it needs internet and
takes a little longer. Expected output:

```
Read 36 resumes from data/resumes
Made 166 section chunks
Loading embedding model all-MiniLM-L6-v2 (the first run downloads it)
Stored 166 chunks in ChromaDB at vector_db/
Ingestion took ... seconds

Example record:
  Name:       Aarav Sharma
  Skills:     Python, Django, PostgreSQL, Redis, REST APIs, AWS
  Experience: 1 year(s)
  ...
```

Run it again whenever you add or change resumes. It rebuilds the collection from
scratch.

## 5. Match job descriptions (Part B)

One job description:

```bash
python3 job_matcher.py data/job_descriptions/jd_01_python_backend.txt
```

All six:

```bash
python3 job_matcher.py --all
```

A pasted job description:

```bash
python3 job_matcher.py --text "Backend developer with 5+ years Python. Django and AWS are a plus."
```

Other options:

| Option | What it does |
|---|---|
| `--top-k 5` | return 5 candidates instead of 10 |
| `--no-filter` | keep candidates who miss a must-have (useful to see what the filter removes) |
| `--llm` | write the reasoning with gpt-4o-mini through OpenRouter (needs the key from step 2) |

Each run prints a short table and saves the full result to
`output/matches_<job name>.json`.

Example of the printed table (your scores will differ a little):

```
Job: Senior Python Backend Engineer
  Must have:    Python, Django
  Nice to have: PostgreSQL, Celery, Docker, AWS
  Min years:    5
  Search took   40 ms
   1. Vikram Menon           score  69  skills: Python, Django, PostgreSQL, Celery, Docker
   2. Meera Kulkarni         score  65  skills: Python, Django, PostgreSQL, AWS
   3. Rohan Reddy            score  58  skills: Python, Django, PostgreSQL, Celery
  Saved output/matches_jd_01_python_backend.json
```

Only 3 names appear because only 3 of the 36 people have Python, Django and at
least 5 years. That is the must-have filter doing its job. A free-text query with
fewer requirements returns the full 10.

## 6. Evaluation

```bash
python3 evaluate.py
```

This prints a summary and writes `output/evaluation_report.md` and
`output/metrics.json`.

## 7. Notebook

```bash
jupyter notebook analysis.ipynb
```

Then use **Run → Run All Cells**. In VS Code, open the file, pick the `.venv`
interpreter as the kernel and click **Run All**. The notebook builds the database
itself if `vector_db/` is missing.

## Common problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'chromadb'` (or any other package) | The virtual environment is not active. Run `source .venv/bin/activate` and try again. |
| `No resume database found. Run: python resume_rag.py` | Step 4 has not been run yet, or it was run from a different folder. Run every command from the project folder. |
| `Folder not found: data/resumes` | Same cause, you are in the wrong folder. `cd` into the project folder. |
| The model download fails or hangs | Check your internet connection. Behind a company proxy, set `HTTPS_PROXY` first. The model is cached after the first download. |
| ChromaDB errors after upgrading the package | Delete the `vector_db/` folder and run `python3 resume_rag.py` again. |
| `--llm` gives the rule-based reasoning anyway | The key is missing or wrong. Check `.env`. The script prints why the API call failed. |
| `The dataset was generated in 2026. Run python generate_dataset.py ...` | The year changed since the dataset was made, so "present" jobs are now a year longer. Run the two commands it names, then `evaluate.py` again. |
| On Apple Silicon, a warning about `tokenizers` and fork | Harmless. You can hide it with `export TOKENIZERS_PARALLELISM=false`. |
