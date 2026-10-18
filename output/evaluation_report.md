# Evaluation report

Dataset: 36 resumes, 166 section chunks, 6 job descriptions.

## 1. Metadata extraction accuracy

| Field | Accuracy |
|---|---|
| name | 100% |
| skills | 100% |
| experience_years | 100% |
| education | 100% |

## 2. Must-have filter

| Job | Kept | Qualified | Precision | Recall |
|---|---|---|---|---|
| jd_01_python_backend | 3 | 3 | 1.00 | 1.00 |
| jd_02_ml_engineer | 4 | 4 | 1.00 | 1.00 |
| jd_03_frontend | 4 | 4 | 1.00 | 1.00 |
| jd_04_devops | 3 | 3 | 1.00 | 1.00 |
| jd_05_data_engineer | 4 | 4 | 1.00 | 1.00 |
| jd_06_java_backend | 3 | 3 | 1.00 | 1.00 |

## 3. Retrieval quality without the filter

| Method | Precision@5 | Recall@10 | MRR |
|---|---|---|---|
| semantic | 0.97 | 1.00 | 0.49 |
| keyword | 1.00 | 1.00 | 0.75 |
| hybrid | 1.00 | 1.00 | 0.60 |

Per job, hybrid:

| Job | Precision@5 | Recall@10 | MRR |
|---|---|---|---|
| jd_01_python_backend | 1.00 | 1.00 | 1.00 |
| jd_02_ml_engineer | 1.00 | 1.00 | 0.50 |
| jd_03_frontend | 1.00 | 1.00 | 0.25 |
| jd_04_devops | 1.00 | 1.00 | 0.50 |
| jd_05_data_engineer | 1.00 | 1.00 | 1.00 |
| jd_06_java_backend | 1.00 | 1.00 | 0.33 |

## 4. Full pipeline (filter + hybrid + score)

Top candidate is a strong match for 100% of the job descriptions.

| Job | Returned | Top candidate | Score | Strong match |
|---|---|---|---|---|
| jd_01_python_backend | 3 | Vikram Menon | 96 | yes |
| jd_02_ml_engineer | 4 | Kavya Joshi | 84 | yes |
| jd_03_frontend | 4 | Rahul Kapoor | 89 | yes |
| jd_04_devops | 3 | Lakshmi Murthy | 98 | yes |
| jd_05_data_engineer | 4 | Joseph Thomas | 91 | yes |
| jd_06_java_backend | 3 | Farhan Qureshi | 95 | yes |

## 5. Speed

- Ingestion of all resumes: 6.45 s (embedding part 0.68 s)
- Search latency over 30 searches: mean 6.3 ms, p95 7.1 ms, max 7.2 ms
