"""Generate the test dataset: 36 resumes (PDF, DOCX, TXT), 6 job descriptions and
the ground truth used by evaluate.py.

Every person, company and university in here is made up. The generator uses a
fixed random seed, so running it twice gives exactly the same files.

    python generate_dataset.py
"""
import datetime
import json
import os
import random
import textwrap

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

random.seed(7)

CURRENT_YEAR = datetime.date.today().year
RESUME_DIR = "data/resumes"
JD_DIR = "data/job_descriptions"

FIRST_NAMES = [
    "Aarav", "Diya", "Rohan", "Ananya", "Vikram", "Meera", "Karthik", "Sneha", "Arjun", "Pooja",
    "Siddharth", "Kavya", "Nikhil", "Ishita", "Rahul", "Tanvi", "Aditya", "Neha", "Manoj", "Ritika",
    "Varun", "Shreya", "Harish", "Lakshmi", "Imran", "Fatima", "Joseph", "Maria", "Gurpreet", "Simran",
    "Abhishek", "Divya", "Suresh", "Anjali", "Farhan", "Nandini",
]
LAST_NAMES = [
    "Sharma", "Iyer", "Reddy", "Nair", "Menon", "Kulkarni", "Patel", "Gupta", "Rao", "Das",
    "Bose", "Joshi", "Verma", "Pillai", "Kapoor", "Mehta", "Singh", "Chopra", "Desai", "Banerjee",
    "Malhotra", "Saxena", "Krishnan", "Murthy", "Khan", "Sheikh", "Thomas", "Fernandes", "Gill", "Kaur",
    "Mishra", "Agarwal", "Naidu", "Bhat", "Qureshi", "Hegde",
]
CITIES = ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Mumbai", "Delhi", "Kochi", "Noida", "Kolkata", "Ahmedabad"]
COMPANIES = [
    "Nimbus Analytics", "Brightline Tech", "Orbit Retail", "Quanta Labs", "CloudHarbor", "FinEdge Systems",
    "Pixelcraft Studios", "VisionStack", "DataWeave Labs", "Northwind Software", "BluePeak Digital",
    "Helix Health Tech", "Arcadia Payments", "Cobalt Logistics", "Lumen EdTech", "Saffron Commerce",
]
UNIVERSITIES = [
    "Pune University", "Anna University", "University of Hyderabad", "Mumbai University",
    "Delhi Technological University", "VIT Vellore", "Manipal Institute of Technology",
    "NIT Trichy", "IIIT Bangalore", "Osmania University",
]

# Each role family: what the person is called, the skills they usually have,
# and sentences used to write their experience.
ROLE_FAMILIES = {
    "python_backend": {
        "titles": ["Backend Engineer", "Python Developer", "Senior Software Engineer"],
        "summary": "Backend engineer with {years} of experience building APIs and data services in Python.",
        "core": ["Python", "Django", "PostgreSQL"],
        "optional": ["FastAPI", "Docker", "AWS", "Redis", "Celery", "REST APIs", "Kafka", "Git"],
        "bullets": [
            "Built REST APIs in Django serving over 2 million requests a day.",
            "Designed PostgreSQL schemas and cut slow query time by 60 percent.",
            "Moved background jobs to Celery and Redis, removing nightly batch delays.",
            "Containerised services with Docker and deployed them on AWS.",
            "Wrote Python services that consume events from Kafka topics.",
        ],
    },
    "ml_engineer": {
        "titles": ["Machine Learning Engineer", "Data Scientist", "Applied Scientist"],
        "summary": "Machine learning engineer with {years} of experience taking models from notebook to production.",
        "core": ["Python", "PyTorch", "Machine Learning"],
        "optional": ["TensorFlow", "scikit-learn", "MLOps", "Docker", "NLP", "Pandas", "SQL", "FastAPI"],
        "bullets": [
            "Trained PyTorch models for product image classification with 94 percent accuracy.",
            "Built an NLP pipeline that tags customer reviews by topic and sentiment.",
            "Set up MLOps pipelines for automated retraining and model monitoring.",
            "Served models behind FastAPI endpoints with sub-100 ms latency.",
            "Built feature pipelines in Pandas and SQL for churn prediction.",
        ],
    },
    "frontend": {
        "titles": ["Frontend Engineer", "UI Developer", "Senior Frontend Developer"],
        "summary": "Frontend developer with {years} of experience building fast, accessible web apps.",
        "core": ["React", "TypeScript", "JavaScript"],
        "optional": ["Next.js", "CSS", "Jest", "Redux", "GraphQL", "Figma", "HTML"],
        "bullets": [
            "Rebuilt the checkout flow in React and TypeScript, cutting load time by 40 percent.",
            "Introduced component tests with Jest and raised coverage to 85 percent.",
            "Moved the marketing site to Next.js with server-side rendering.",
            "Connected UI screens to a GraphQL API and managed state with Redux.",
            "Turned Figma designs into responsive pages with modern CSS.",
        ],
    },
    "devops": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"],
        "summary": "DevOps engineer with {years} of experience running cloud infrastructure at scale.",
        "core": ["Kubernetes", "Terraform", "AWS"],
        "optional": ["Docker", "CI/CD", "Linux", "Python", "Azure", "Prometheus", "Ansible", "Bash"],
        "bullets": [
            "Ran Kubernetes clusters hosting more than 200 microservices.",
            "Wrote Terraform modules that provision AWS environments in minutes.",
            "Built CI/CD pipelines that cut release time from days to hours.",
            "Set up Prometheus alerts that reduced on-call incidents by a third.",
            "Automated cost reports across AWS accounts with Python scripts.",
        ],
    },
    "data_engineer": {
        "titles": ["Data Engineer", "Big Data Engineer", "Analytics Engineer"],
        "summary": "Data engineer with {years} of experience building reliable batch and streaming pipelines.",
        "core": ["Spark", "SQL", "Python"],
        "optional": ["Kafka", "Airflow", "AWS", "Snowflake", "dbt", "Hadoop", "Scala"],
        "bullets": [
            "Built Spark jobs that process 5 TB of clickstream data every day.",
            "Scheduled and monitored pipelines with Airflow.",
            "Streamed order events through Kafka into the data warehouse.",
            "Modelled analytics tables in Snowflake using dbt.",
            "Rewrote slow SQL reports, bringing runtime down from hours to minutes.",
        ],
    },
    "java_backend": {
        "titles": ["Java Developer", "Backend Engineer", "Senior Java Engineer"],
        "summary": "Java developer with {years} of experience building backend microservices.",
        "core": ["Java", "Spring Boot", "Microservices"],
        "optional": ["Kafka", "MySQL", "Hibernate", "Docker", "Kubernetes", "Redis", "AWS"],
        "bullets": [
            "Built Spring Boot payment services handling 50,000 transactions a day.",
            "Split a monolith into microservices with clear API contracts.",
            "Added Kafka event streaming for order status updates.",
            "Tuned Hibernate mappings and MySQL indexes for faster reports.",
            "Deployed services as Docker containers on Kubernetes.",
        ],
    },
}

# Used when a person has run out of role-specific lines
GENERIC_BULLETS = [
    "Reviewed pull requests and mentored two junior engineers.",
    "Wrote technical design documents and ran knowledge-sharing sessions.",
    "Worked with product managers to plan quarterly releases.",
]

# Some resumes write a skill differently. The matcher has to understand these.
ALIASES = {"PostgreSQL": "Postgres", "Kubernetes": "K8s", "JavaScript": "JS", "Machine Learning": "ML"}

# Different people use different section titles. The chunker has to handle all of them.
HEADINGS = {
    "summary": ["SUMMARY", "PROFILE", "PROFESSIONAL SUMMARY"],
    "skills": ["SKILLS", "TECHNICAL SKILLS", "CORE SKILLS"],
    "experience": ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE"],
    "projects": ["PROJECTS", "KEY PROJECTS"],
    "education": ["EDUCATION", "ACADEMIC BACKGROUND"],
    "certifications": ["CERTIFICATIONS"],
}

# Years of experience for the 6 people in each family. Spread out on purpose so
# that some people pass the minimum experience of a job and some do not.
YEARS_PLAN = [1, 3, 5, 6, 8, 11]

JOB_DESCRIPTIONS = [
    {
        "id": "jd_01_python_backend", "family": "python_backend",
        "title": "Senior Python Backend Engineer",
        "must_have": ["Python", "Django"], "nice_to_have": ["PostgreSQL", "Docker", "AWS", "Celery"],
        "min_years": 5,
        "description": "We are looking for a backend engineer to build and scale the APIs behind our "
                       "e-commerce platform. You will design database schemas, write Django services, "
                       "own background job processing and help us move more workloads to the cloud.",
    },
    {
        "id": "jd_02_ml_engineer", "family": "ml_engineer",
        "title": "Machine Learning Engineer",
        "must_have": ["Python", "PyTorch"], "nice_to_have": ["MLOps", "NLP", "Docker", "TensorFlow"],
        "min_years": 3,
        "description": "Join our applied AI team to train, deploy and monitor models in production. "
                       "You will work on recommendation and text classification problems and build the "
                       "pipelines that retrain models automatically.",
    },
    {
        "id": "jd_03_frontend", "family": "frontend",
        "title": "Frontend Developer (React)",
        "must_have": ["React", "TypeScript"], "nice_to_have": ["Next.js", "Jest", "CSS", "GraphQL"],
        "min_years": 2,
        "description": "Build the customer-facing web app used by millions of shoppers. You will turn "
                       "designs into fast, accessible screens, write component tests and improve page "
                       "load performance.",
    },
    {
        "id": "jd_04_devops", "family": "devops",
        "title": "DevOps Engineer",
        "must_have": ["Kubernetes", "Terraform"], "nice_to_have": ["AWS", "CI/CD", "Prometheus", "Python"],
        "min_years": 4,
        "description": "Own our cloud infrastructure end to end. You will manage container platforms, "
                       "write infrastructure as code, improve release pipelines and keep production "
                       "reliable with good monitoring.",
    },
    {
        "id": "jd_05_data_engineer", "family": "data_engineer",
        "title": "Data Engineer",
        "must_have": ["Spark", "SQL"], "nice_to_have": ["Kafka", "Airflow", "Snowflake", "dbt"],
        "min_years": 3,
        "description": "Design and run the data pipelines that feed our analytics and reporting. You will "
                       "process large event streams, schedule batch jobs and model warehouse tables "
                       "for analysts.",
    },
    {
        "id": "jd_06_java_backend", "family": "java_backend",
        "title": "Java Backend Engineer",
        "must_have": ["Java", "Spring Boot"], "nice_to_have": ["Kafka", "Microservices", "MySQL", "Docker"],
        "min_years": 5,
        "description": "Build high-volume payment and order services. You will design microservices, "
                       "work with event streaming and relational databases, and keep services fast "
                       "and reliable under load.",
    },
]


def years_text(years):
    return "1 year" if years == 1 else f"{years} years"


def make_candidate(index, family_name, years, drop_core_skill):
    """Build one candidate as plain data: name, skills, jobs, education."""
    family = ROLE_FAMILIES[family_name]
    first, last = FIRST_NAMES[index], LAST_NAMES[index]

    skills = list(family["core"])
    bullets = list(family["bullets"])
    if drop_core_skill:
        # One person per family is missing a key skill, so the must-have filter has real work to do.
        # Their experience lines must not mention that skill either.
        dropped = family["core"][1]
        skills.remove(dropped)
        bullets = [line for line in bullets if dropped.lower() not in line.lower()]
    skills += random.sample(family["optional"], k=random.randint(2, 4))

    # Keep only experience lines whose tools are all in this person's skills list.
    # Otherwise a line could mention Docker while the skills list does not.
    family_skills = family["core"] + family["optional"]
    usable = []
    for line in bullets:
        mentioned = [skill for skill in family_skills if skill.lower() in line.lower()]
        if all(skill in skills for skill in mentioned):
            usable.append(line)
    bullets = usable

    # Split the career into 1 to 3 jobs that end in the current year
    job_count = 1 if years <= 2 else random.randint(2, 3)
    cut_points = sorted(random.sample(range(1, years), k=job_count - 1)) if job_count > 1 else []
    boundaries = [0] + cut_points + [years]
    companies = random.sample(COMPANIES, k=job_count)

    # Shuffle the experience lines once and hand them out, so two jobs do not
    # repeat the same line. The latest job gets the first two.
    random.shuffle(bullets)

    jobs = []
    for job_number in range(job_count):
        start = CURRENT_YEAR - years + boundaries[job_number]
        end = CURRENT_YEAR - years + boundaries[job_number + 1]
        is_latest = job_number == job_count - 1
        jobs.append({
            "title": random.choice(family["titles"]),
            "company": companies[job_number],
            "start": start,
            "end": "present" if is_latest else end,
            "bullets": [],
        })
    jobs.reverse()   # newest job first, like a real resume
    for job in jobs:
        take = 2 if len(bullets) >= 2 else 1
        for _ in range(take):
            if bullets:
                job["bullets"].append(bullets.pop(0))
            else:
                job["bullets"].append(random.choice(GENERIC_BULLETS))

    graduation = CURRENT_YEAR - years
    degree = random.choice(["B.Tech in Computer Science", "B.E. in Information Technology",
                            "M.Tech in Computer Science", "B.Sc. in Mathematics", "MCA"])
    return {
        "file_stem": f"{first.lower()}_{last.lower()}",
        "name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "city": random.choice(CITIES),
        "family": family_name,
        "years": years,
        "summary": family["summary"].format(years=years_text(years)),
        "skills": skills,
        "jobs": jobs,
        "education": f"{degree}, {random.choice(UNIVERSITIES)}, {graduation}",
        "has_projects": random.random() < 0.4,
        "has_certifications": random.random() < 0.3,
    }


def resume_text(candidate):
    """Write the candidate out as resume text, with varied headings and some skill aliases."""
    def heading(section):
        return random.choice(HEADINGS[section])

    written_skills = []
    for skill in candidate["skills"]:
        if skill in ALIASES and random.random() < 0.4:
            written_skills.append(ALIASES[skill])
        else:
            written_skills.append(skill)

    lines = [candidate["name"], f"{candidate['email']} | {candidate['city']}, India", ""]
    lines += [heading("summary"), candidate["summary"], ""]
    lines += [heading("skills"), ", ".join(written_skills), ""]
    lines.append(heading("experience"))
    for job in candidate["jobs"]:
        lines.append(f"{job['title']}, {job['company']} ({job['start']} - {job['end']})")
        for bullet in job["bullets"]:
            lines.append(f"- {bullet}")
        lines.append("")
    if candidate["has_projects"]:
        lines += [heading("projects"),
                  f"- Contributed bug fixes to an open-source {random.choice(candidate['skills'])} library.",
                  "- Built a personal budgeting app used by friends and family.", ""]
    lines += [heading("education"), candidate["education"], ""]
    if candidate["has_certifications"]:
        lines += [heading("certifications"),
                  random.choice(["AWS Certified Developer - Associate", "Certified Kubernetes Administrator",
                                 "Google Professional Data Engineer", "Oracle Certified Java Programmer",
                                 "TensorFlow Developer Certificate"]), ""]
    return "\n".join(lines).strip() + "\n"


def save_pdf(path, text):
    pdf = canvas.Canvas(path, pagesize=A4)
    y = 800
    for paragraph in text.split("\n"):
        # PDF lines do not wrap by themselves, so long lines are wrapped here
        for line in textwrap.wrap(paragraph, width=95) or [""]:
            if y < 60:          # start a new page near the bottom
                pdf.showPage()
                y = 800
            pdf.drawString(50, y, line)
            y -= 15
    pdf.save()


def save_docx(path, text):
    document = Document()
    for line in text.split("\n"):
        document.add_paragraph(line)
    document.save(path)


def main():
    os.makedirs(RESUME_DIR, exist_ok=True)
    os.makedirs(JD_DIR, exist_ok=True)

    candidates = []
    index = 0
    for family_name in ROLE_FAMILIES:
        for years in YEARS_PLAN:
            # The person with 6 years in every family is missing one core skill
            candidate = make_candidate(index, family_name, years, drop_core_skill=(years == 6))
            candidates.append(candidate)
            index += 1

    formats = ["pdf", "docx", "txt"]
    for number, candidate in enumerate(candidates):
        file_type = formats[number % 3]
        path = f"{RESUME_DIR}/{candidate['file_stem']}.{file_type}"
        text = resume_text(candidate)
        if file_type == "pdf":
            save_pdf(path, text)
        elif file_type == "docx":
            save_docx(path, text)
        else:
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
        candidate["resume_path"] = path

    # Job description files in a simple, readable format
    for jd in JOB_DESCRIPTIONS:
        content = (
            f"Title: {jd['title']}\n"
            f"Must have: {', '.join(jd['must_have'])}\n"
            f"Nice to have: {', '.join(jd['nice_to_have'])}\n"
            f"Minimum experience: {jd['min_years']} years\n\n"
            f"About the role:\n{jd['description']}\n"
        )
        with open(f"{JD_DIR}/{jd['id']}.txt", "w", encoding="utf-8") as file:
            file.write(content)

    # Ground truth for evaluate.py, based on the true data, not on what the
    # pipeline extracts from the files.
    #   qualified: has every must-have skill and enough years
    #   strong:    qualified, same role family, and at least 2 of the nice-to-have skills
    ground_truth = {}
    for jd in JOB_DESCRIPTIONS:
        qualified, strong = [], []
        for candidate in candidates:
            has_must = all(skill in candidate["skills"] for skill in jd["must_have"])
            if has_must and candidate["years"] >= jd["min_years"]:
                qualified.append(candidate["resume_path"])
                nice_count = sum(1 for skill in jd["nice_to_have"] if skill in candidate["skills"])
                if candidate["family"] == jd["family"] and nice_count >= 2:
                    strong.append(candidate["resume_path"])
        ground_truth[jd["id"]] = {"qualified": qualified, "strong": strong}

    truth_records = []
    for candidate in candidates:
        truth_records.append({
            "resume_path": candidate["resume_path"],
            "name": candidate["name"],
            "family": candidate["family"],
            "experience_years": candidate["years"],
            "skills": candidate["skills"],
            "education": candidate["education"],
        })

    with open("data/ground_truth.json", "w", encoding="utf-8") as file:
        # The year matters because "2019 - present" means a different number of years next year
        json.dump({"generated_in_year": CURRENT_YEAR, "jobs": ground_truth, "candidates": truth_records}, file, indent=2)

    print(f"Created {len(candidates)} resumes in {RESUME_DIR}")
    print(f"Created {len(JOB_DESCRIPTIONS)} job descriptions in {JD_DIR}")
    for jd in JOB_DESCRIPTIONS:
        truth = ground_truth[jd["id"]]
        print(f"  {jd['id']}: {len(truth['qualified'])} qualified, {len(truth['strong'])} strong matches")


if __name__ == "__main__":
    main()
