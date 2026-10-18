# RAG Based Profile Matching

RAG system that reads a pile of resumes,
stores them in a vector database and, given a job description, returns the 10
candidates who fit best, with a score, the skills that matched, the resume
sections that matched and a short reason for each.

Setup and run steps are in [SETUP.md](SETUP.md). This file explains what the
project does and why it is built this way.

## What is in the folder

```
rag-based-profile-matching/
├── README.md               this file
├── SETUP.md                install and run steps, expected output, common errors
├── requirements.txt
├── .env.example            only needed for the optional --llm reasoning
├── fs_tools.py             file tools from Milestone 1 (read PDF, DOCX, TXT, list files)
├── generate_dataset.py     creates the 36 resumes, 6 job descriptions and the ground truth
├── resume_rag.py           Part A: read, chunk, extract metadata, embed, store in ChromaDB
├── job_matcher.py          Part B: filter, hybrid search, score, reasoning, JSON output
├── evaluate.py             retrieval accuracy and latency, writes the evaluation report
├── analysis.ipynb          experiments and charts
├── docs/
│   └── DEMO_SCRIPT.md      what I show in the 3 to 4 minute demo video
├── data/
│   ├── resumes/            36 resumes (12 PDF, 12 DOCX, 12 TXT)
│   ├── job_descriptions/   6 job descriptions
│   └── ground_truth.json   the right answers, used by evaluate.py
├── vector_db/              created by resume_rag.py (ChromaDB files + candidates.json)
└── output/                 match results, metrics and the evaluation report
```

## How it works

```
 INGESTION (resume_rag.py, run once)

 data/resumes/*.pdf|docx|txt
        │  list_files + read_file  (Milestone 1 tools)
        ▼
 plain text of each resume
        │  split at section headings
        ▼
 summary | skills | experience | projects | education | certifications
        │                                   │
        │ embed each section                │ extract metadata
        │ (all-MiniLM-L6-v2, 384 numbers)   │ name, skills, years, education
        ▼                                   ▼
 ChromaDB collection "resumes"  ◄───────────┘  (metadata stored on every chunk)
 + vector_db/candidates.json


 MATCHING (job_matcher.py, once per job description)

 job description
        │  parse: title, must-have, nice-to-have, minimum years
        ▼
 must-have filter ──► only candidates with every must-have skill and enough years
        │
        ├──► semantic search in ChromaDB (same filter as a "where" clause)
        │       best section similarity per candidate
        │
        ├──► keyword search (BM25 over resume words + skill tokens,
        │       must-have skills counted twice)
        ▼
 Reciprocal Rank Fusion  ──►  top 10
        │
        ▼
 score 0-100, matched skills, excerpts, matched sections, reasoning
        │
        ▼
 output/matches_<job>.json
```

### Part A: building the resume database

**Reading.** `resume_rag.py` uses `list_files` and `read_file` from Milestone 1, so
PDF, DOCX and TXT all come back as plain text. Files that fail to open or have no
text are skipped with a message instead of stopping the run.

**Chunking by section.** Each resume is split at its headings. People write
headings differently (SUMMARY, PROFILE, PROFESSIONAL SUMMARY, WORK EXPERIENCE,
ACADEMIC BACKGROUND ...), so there is a lookup table that maps all of them to six
standard names. One section becomes one chunk. If a section is longer than 800
characters it is cut between lines, and each piece repeats the last line of the
previous piece, so a job title never ends up in a different chunk from its first
bullet point.

I chose this over fixed-size chunks because a fixed 500-character window would
happily start halfway through one job and end halfway through the next. With
sections, every excerpt the matcher shows is a complete piece of the resume, and
the section name tells the recruiter where the match came from.

Each chunk starts with a small header line, for example `Arjun Rao | experience`,
so the embedding knows whose resume and which part it is.

**Metadata.** For every resume the script pulls out:

| Field | How |
|---|---|
| Name | first line of the resume |
| Skills | the skills section is checked against a list of about 50 known skills. Different spellings map to one name (K8s → Kubernetes, Postgres → PostgreSQL, ML → Machine Learning). Matching is whole-word only, so "java" does not match inside "javascript". |
| Experience years | from the date ranges in the experience section, for example `(2019 - present)`: current year minus the earliest start year. If there are no dates, it falls back to a sentence like "8 years of experience". |
| Education | first line of the education section |

ChromaDB only accepts plain text, numbers and True/False as metadata, not lists.
So skills are stored in two ways on every chunk: as readable text
(`"skills": "Python, Django, AWS"`) and as one flag per skill
(`"has_django": true`). The flags are what make filtering inside ChromaDB possible.

**Embeddings and storage.** Sections are embedded with `all-MiniLM-L6-v2` from
sentence-transformers. It is free, runs on a laptop and gives 384 numbers per text.
The vectors are normalized and stored in a persistent ChromaDB collection that uses
cosine distance. The script deletes the old collection before adding, so running it
twice does not create duplicates.

### Part B: matching a job description

**Parsing the job description.** The job description files use labelled lines:

```
Title: Senior Python Backend Engineer
Must have: Python, Django
Nice to have: PostgreSQL, Docker, AWS, Celery
Minimum experience: 5 years
```

A pasted free-text job description also works. Then the parser looks for patterns
like "5+ years Python" (Python becomes a must-have, 5 the minimum) and treats every
other known skill it finds as nice to have.

**Must-have filter.** A candidate who lacks a must-have skill, or has fewer years
than the minimum, is removed before ranking. The same rule is passed to ChromaDB as
a `where` clause, for example:

```python
{"$and": [{"has_python": True}, {"has_django": True}, {"experience_years": {"$gte": 5}}]}
```

Embeddings are good at meaning but they cannot reliably tell 4 years from 6, and
they do not know that Django is required rather than just nice. Hard requirements
are rules, so they are handled as rules.

**Hybrid search.** Two rankings are made for the candidates who pass the filter:

- *Semantic*: the job description is embedded and compared with every section.
  Each candidate is ranked by their closest section.
- *Keyword*: BM25 over the resume words plus a special token for each known skill
  (`skill_kubernetes`). The token is the same whether the resume says Kubernetes or
  K8s. Must-have skills appear twice in the query, so they count more than
  nice-to-haves.

The two rankings are combined with Reciprocal Rank Fusion: each candidate gets
`1 / (60 + rank)` from each list and the totals are sorted. RRF only uses rank
positions, so I never have to add a BM25 score (which can be 5.3) to a cosine
similarity (which is between -1 and 1). The top 10 after fusion are kept.

**Score from 0 to 100.**

| Part | Points | How |
|---|---|---|
| Meaning | 45 | best section similarity, stretched from the range 0.20–0.65 to 0–1 |
| Skills | 35 | share of the job's skills the candidate lists, must-haves count double |
| Experience | 20 | years ÷ minimum, capped at 1.5 times the minimum. Meeting the minimum exactly gives 13 of 20. |

RRF decides who makes the top 10. The score decides the order they are shown in,
so the numbers in the output always go from high to low and every point can be
explained.

**Reasoning.** By default the reason is built from the facts: years against the
minimum, must-have skills, which nice-to-have skills are there and which are
missing, the sections that matched and the closest similarity. With `--llm` the same
facts are sent to `gpt-4o-mini` through OpenRouter, which writes a two-sentence note.
If the API call fails, the rule-based reason is used, so the matcher never breaks
because of the network.

### Output format

```json
{
  "job_description": "Title: Senior Python Backend Engineer ...",
  "top_matches": [
    {
      "candidate_name": "Vikram Menon",
      "resume_path": "data/resumes/vikram_menon.docx",
      "match_score": 69,
      "matched_skills": ["Python", "Django", "PostgreSQL", "Celery", "Docker"],
      "relevant_excerpts": [
        "[summary] Backend engineer with 8 years of experience building APIs ...",
        "[experience] Senior Software Engineer, DataWeave Labs (2019 - present) ..."
      ],
      "reasoning": "8 years of experience against the 5+ asked for. Has every must-have skill (Python, Django). Covers 3 of 4 nice-to-have skills (PostgreSQL, Celery, Docker), without AWS. Sections that matched the job: summary, experience, skills. ..."
    }
  ]
}
```

The exact names and numbers depend on your run.

## Dataset

Real resumes come with personal data, so I generated the dataset instead.
`generate_dataset.py` writes 36 resumes across six kinds of role (Python backend,
ML, frontend, DevOps, data engineering, Java backend), six people per role, with
1, 3, 5, 6, 8 and 11 years of experience. It uses a fixed random seed, so it always
produces the same files.

To make it a fair test, the generator varies things on purpose:

- file type rotates between PDF, DOCX and TXT
- section headings change between resumes
- some skills use short spellings (K8s, Postgres, ML, JS)
- in every role, the person with 6 years is missing one key skill, so they read like
  a good fit but should be removed by the must-have filter
- some resumes have projects or certifications sections, some do not

There are 6 job descriptions, one per role, each with 2 must-have skills, 4
nice-to-have skills and a minimum of 2 to 5 years.

Because the data is generated, the right answers are known. `ground_truth.json`
lists, for each job, the **qualified** candidates (every must-have and enough years)
and the **strong** matches (qualified, same role, and at least 2 nice-to-have skills).

## Evaluation

`python evaluate.py` writes `output/evaluation_report.md` and `output/metrics.json`.

| What | Metric |
|---|---|
| Metadata extraction | accuracy of name, skills, years and education against the ground truth |
| Must-have filter | precision and recall of the kept candidates against the qualified set |
| Retrieval, with the filter switched off | Precision@5 (share of the top 5 in the right role), Recall@10 (qualified candidates found in the top 10), MRR (1 ÷ rank of the first strong match) for semantic, keyword and hybrid |
| Full pipeline | how often the top candidate is a strong match |
| Speed | ingestion time, and mean and p95 search latency over 30 searches |

The retrieval comparison runs without the filter on purpose. With the filter on,
only 3 or 4 people are left for each job and every method looks perfect, which
says nothing about the search itself.

`analysis.ipynb` repeats these with charts and adds a few experiments: what the
filter changes, whether K8s and Kubernetes give the same people, the effect of the
RRF constant, and the breakdown of the score.

## Design choices and trade-offs

- **ChromaDB** because it runs locally with no account, keeps the data on disk and
  supports metadata filters. Pinecone or Weaviate would need a cloud account for
  something that fits in a few megabytes.
- **all-MiniLM-L6-v2** because it is small and free. A larger model would probably
  separate similar roles better, at the cost of speed.
- **Rules for metadata, not an LLM.** Rules are free, fast and give the same answer
  every time. An LLM would handle messy real resumes better but adds cost and can
  make things up.
- **No cross-encoder re-ranking.** With 36 resumes the filter leaves only a handful
  of people per job, so a re-ranker would add download size and time for very little
  change. For thousands of resumes it would be worth adding after RRF.

## Limitations

- The dataset is generated, so it is cleaner than real life. Real resumes have
  tables, two-column layouts, scanned pages and headings that are not on the list.
- Skills are only read from the skills section. A skill mentioned only in an
  experience bullet is not counted as a listed skill (BM25 still sees it).
- The skill list is fixed. A skill that is not on it is invisible to the filter and
  the skill score.
- Years of experience are counted from the first start date to the last end date, so
  gaps between jobs are not subtracted.
- The score weights (45 / 35 / 20) are my choice, not learned from hiring data.

## Ideas for later

- Read skills from the experience section too, and weight recent experience higher
- Add a cross-encoder re-ranking step for larger pools
- Let a recruiter adjust the score weights
- A small Streamlit page to paste a job description and browse the results
