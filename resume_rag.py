"""RAG Based Profile Matching, Part A: resume ingestion pipeline.

What this script does, in order:
  1. Finds every resume in data/resumes and reads it with the Milestone 1 tools (fs_tools.py)
  2. Splits each resume into its sections (Summary, Skills, Experience, ...)
  3. Pulls out metadata: name, skills, years of experience, education
  4. Turns every section into an embedding (a list of 384 numbers)
  5. Stores the sections, embeddings and metadata in ChromaDB (folder vector_db/)
  6. Saves one summary record per candidate in vector_db/candidates.json

Run it once before job_matcher.py, and again whenever the resumes change:

    python resume_rag.py
"""
import datetime
import json
import os
import re
import time

import chromadb
from sentence_transformers import SentenceTransformer

from fs_tools import list_files, read_file

RESUME_FOLDER = "data/resumes"
DB_FOLDER = "vector_db"
COLLECTION_NAME = "resumes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, free, runs on a laptop, 384 numbers per text

MAX_CHUNK_CHARS = 800      # a section longer than this is split into smaller pieces
OVERLAP_LINES = 1          # how many lines a piece repeats from the piece before it

CURRENT_YEAR = datetime.date.today().year


# ---------------------------------------------------------------------------
# Section headings
# ---------------------------------------------------------------------------
# People name their sections differently. All of these spellings are mapped
# to one standard name, so "WORK EXPERIENCE" and "EXPERIENCE" end up the same.
SECTION_HEADINGS = {
    "SUMMARY": "summary",
    "PROFILE": "summary",
    "PROFESSIONAL SUMMARY": "summary",
    "OBJECTIVE": "summary",
    "SKILLS": "skills",
    "TECHNICAL SKILLS": "skills",
    "CORE SKILLS": "skills",
    "EXPERIENCE": "experience",
    "WORK EXPERIENCE": "experience",
    "PROFESSIONAL EXPERIENCE": "experience",
    "EMPLOYMENT HISTORY": "experience",
    "PROJECTS": "projects",
    "KEY PROJECTS": "projects",
    "EDUCATION": "education",
    "ACADEMIC BACKGROUND": "education",
    "CERTIFICATIONS": "certifications",
}


# ---------------------------------------------------------------------------
# Skill list
# ---------------------------------------------------------------------------
# Each standard skill name, followed by the ways people write it.
# This list is shared with job_matcher.py, so a job description that says
# "K8s" and a resume that says "Kubernetes" are treated as the same skill.
KNOWN_SKILLS = {
    "Python": ["python"],
    "Django": ["django"],
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "SQL": ["sql"],
    "Redis": ["redis"],
    "Celery": ["celery"],
    "REST APIs": ["rest apis", "rest api"],
    "Kafka": ["kafka"],
    "Git": ["git"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "AWS": ["aws"],
    "Azure": ["azure"],
    "CI/CD": ["ci/cd", "cicd"],
    "Linux": ["linux"],
    "Bash": ["bash"],
    "Ansible": ["ansible"],
    "Prometheus": ["prometheus"],
    "Machine Learning": ["machine learning", "ml"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "MLOps": ["mlops"],
    "NLP": ["nlp"],
    "Pandas": ["pandas"],
    "Spark": ["spark", "pyspark"],
    "Airflow": ["airflow"],
    "Snowflake": ["snowflake"],
    "dbt": ["dbt"],
    "Hadoop": ["hadoop"],
    "Scala": ["scala"],
    "Java": ["java"],
    "Spring Boot": ["spring boot"],
    "Microservices": ["microservices"],
    "Hibernate": ["hibernate"],
    "React": ["react", "react.js", "reactjs"],
    "TypeScript": ["typescript"],
    "JavaScript": ["javascript", "js"],
    "Next.js": ["next.js", "nextjs"],
    "CSS": ["css"],
    "HTML": ["html"],
    "Jest": ["jest"],
    "Redux": ["redux"],
    "GraphQL": ["graphql"],
    "Figma": ["figma"],
}


def find_skills(text):
    """Return the standard names of every known skill that appears in the text."""
    text = text.lower()
    found = []
    for skill_name in KNOWN_SKILLS:
        for spelling in KNOWN_SKILLS[skill_name]:
            # Match the spelling only as a whole word. Without this check, "java"
            # would match inside "javascript" and "sql" inside "postgresql".
            # The dot stops "js" from matching the end of "next.js".
            pattern = r"(?<![a-z0-9.])" + re.escape(spelling) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                found.append(skill_name)
                break      # one spelling is enough, go to the next skill
    return found


def skill_key(skill_name):
    """Turn a skill name into a metadata key, for example 'Spring Boot' -> 'has_spring_boot'."""
    return "has_" + re.sub(r"[^a-z0-9]+", "_", skill_name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Step 2: split a resume into sections
# ---------------------------------------------------------------------------
def split_into_sections(text):
    """Return a list of (section_name, section_text) in the order they appear.

    Everything before the first heading (name, email, city) is the "contact" section.
    """
    sections = []
    current_name = "contact"
    current_lines = []

    for line in text.split("\n"):
        heading = line.strip().rstrip(":").upper()
        if heading in SECTION_HEADINGS:
            # A new heading: save the section we were building and start a new one
            if current_lines:
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = SECTION_HEADINGS[heading]
            current_lines = []
        elif line.strip() != "":
            current_lines.append(line.strip())

    if current_lines:
        sections.append((current_name, "\n".join(current_lines).strip()))

    return sections


def split_long_section(section_text):
    """Cut a long section into pieces of at most MAX_CHUNK_CHARS, without breaking lines.

    Most resume sections are short and come back as one piece. A long experience
    section is cut between lines, and each piece repeats the last line of the
    piece before it, so a job title is not separated from its first bullet point.
    """
    if len(section_text) <= MAX_CHUNK_CHARS:
        return [section_text]

    lines = section_text.split("\n")
    pieces = []
    current = []
    for line in lines:
        if current and len("\n".join(current + [line])) > MAX_CHUNK_CHARS:
            pieces.append("\n".join(current))
            current = current[-OVERLAP_LINES:]     # carry the last line over
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


# ---------------------------------------------------------------------------
# Step 3: metadata
# ---------------------------------------------------------------------------
def find_experience_years(sections):
    """Work out total years of experience.

    First choice: the date ranges in the experience section, for example
    "(2019 - present)". Years = current year minus the earliest start year.
    Second choice: a sentence like "8 years of experience" in the summary.
    """
    experience_text = ""
    summary_text = ""
    for name, text in sections:
        if name == "experience":
            experience_text = experience_text + "\n" + text
        if name == "summary":
            summary_text = summary_text + "\n" + text

    # Finds "2019 - present", "2018 - 2021", "2018 – 2021"
    ranges = re.findall(r"((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2}|present|current)", experience_text, re.IGNORECASE)
    if ranges:
        start_years = []
        end_years = []
        for start, end in ranges:
            start_years.append(int(start))
            if end.lower() in ["present", "current"]:
                end_years.append(CURRENT_YEAR)
            else:
                end_years.append(int(end))
        return max(end_years) - min(start_years)

    match = re.search(r"(\d+)\+?\s*years?", summary_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return 0       # unknown


def extract_metadata(text, sections):
    """Name, skills, years of experience and education for one resume."""
    lines = []
    for line in text.split("\n"):
        if line.strip() != "":
            lines.append(line.strip())
    name = lines[0] if lines else "Unknown"     # the name is the first line of the resume

    # Skills come from the skills section. If a resume has no skills section,
    # fall back to searching the whole resume.
    skills_text = ""
    for section_name, section_text in sections:
        if section_name == "skills":
            skills_text = skills_text + " " + section_text
    if skills_text == "":
        skills_text = text
    skills = find_skills(skills_text)

    education = ""
    for section_name, section_text in sections:
        if section_name == "education":
            education = section_text.split("\n")[0]    # the first line holds the main degree
            break

    return {
        "candidate_name": name,
        "skills": skills,
        "experience_years": find_experience_years(sections),
        "education": education,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

    # 1. Find and read every resume with the Milestone 1 tools
    files = list_files(RESUME_FOLDER)
    if files and "error" in files[0]:
        print(files[0]["error"], "- run python generate_dataset.py first.")
        return

    resumes = []
    for file_info in files:
        result = read_file(file_info["path"])
        if not result["success"]:
            print("Skipped", file_info["name"], ":", result["error"])
            continue
        if result["content"] == "":
            print("Skipped", file_info["name"], ": no text found")
            continue
        resumes.append({"path": file_info["path"], "text": result["content"]})
    print("Read", len(resumes), "resumes from", RESUME_FOLDER)

    # 2 and 3. Sections, chunks and metadata
    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []
    candidates = []

    for resume in resumes:
        sections = split_into_sections(resume["text"])
        info = extract_metadata(resume["text"], sections)

        # ChromaDB metadata values must be plain text, numbers or True/False (no lists).
        # So the skills are stored twice:
        #   "skills": "Python, Django, AWS"   readable text
        #   "has_python": True                 one flag per skill, used for filtering
        base_metadata = {
            "candidate_name": info["candidate_name"],
            "resume_path": resume["path"],
            "skills": ", ".join(info["skills"]),
            "experience_years": info["experience_years"],
            "education": info["education"],
        }
        for skill in info["skills"]:
            base_metadata[skill_key(skill)] = True

        section_names = []
        for section_name, section_text in sections:
            if section_name == "contact":
                continue      # name and email do not help with matching
            section_names.append(section_name)

            pieces = split_long_section(section_text)
            for piece_number in range(len(pieces)):
                # Putting the name and section at the top of each chunk gives the
                # embedding some context, for example "Arjun Rao | experience".
                chunk_text = info["candidate_name"] + " | " + section_name + "\n" + pieces[piece_number]

                metadata = dict(base_metadata)        # a copy, so every chunk gets its own
                metadata["section"] = section_name
                metadata["chunk_number"] = piece_number

                file_stem = os.path.splitext(os.path.basename(resume["path"]))[0]
                chunk_ids.append(file_stem + "_" + section_name + "_" + str(piece_number))
                chunk_texts.append(chunk_text)
                chunk_metadatas.append(metadata)

        candidates.append({
            "candidate_name": info["candidate_name"],
            "resume_path": resume["path"],
            "skills": info["skills"],
            "experience_years": info["experience_years"],
            "education": info["education"],
            "sections": section_names,
            "full_text": resume["text"],
        })

    print("Made", len(chunk_texts), "section chunks")

    # 4. Embeddings
    print("Loading embedding model", EMBEDDING_MODEL, "(the first run downloads it)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embed_start = time.time()
    # normalize_embeddings=True makes every vector length 1, so cosine similarity is a plain dot product
    embeddings = model.encode(chunk_texts, normalize_embeddings=True)
    embed_seconds = time.time() - embed_start

    # 5. Store in ChromaDB. The old collection is deleted first, so running
    # the script twice does not store everything twice.
    client = chromadb.PersistentClient(path=DB_FOLDER)
    existing_names = []
    for collection in client.list_collections():
        # Newer ChromaDB returns names, older versions return collection objects
        existing_names.append(collection if isinstance(collection, str) else collection.name)
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,                # we make the embeddings ourselves
        metadata={"hnsw:space": "cosine"},      # compare vectors with cosine distance
    )
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings.tolist(),
        documents=chunk_texts,
        metadatas=chunk_metadatas,
    )

    # 6. Candidate records for keyword search and evaluation
    with open(os.path.join(DB_FOLDER, "candidates.json"), "w", encoding="utf-8") as file:
        json.dump(candidates, file, indent=2)

    total_seconds = time.time() - start_time
    stats = {
        "resumes": len(candidates),
        "chunks": len(chunk_texts),
        "embedding_seconds": round(embed_seconds, 2),
        "total_ingestion_seconds": round(total_seconds, 2),
    }
    os.makedirs("output", exist_ok=True)
    with open("output/ingestion_stats.json", "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print("Stored", collection.count(), "chunks in ChromaDB at", DB_FOLDER + "/")
    print("Ingestion took", round(total_seconds, 2), "seconds")
    print("\nExample record:")
    example = candidates[0]
    print("  Name:      ", example["candidate_name"])
    print("  Skills:    ", ", ".join(example["skills"]))
    print("  Experience:", example["experience_years"], "year(s)")
    print("  Education: ", example["education"])
    print("  Sections:  ", ", ".join(example["sections"]))


if __name__ == "__main__":
    main()
