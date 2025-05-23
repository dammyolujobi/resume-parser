# Resume Parser and Matching API

A complete FastAPI-based system for parsing resumes, extracting key information using NLP, and matching candidates to job profiles.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Configuration](#configuration)
5. [Installing Dependencies](#installing-dependencies)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [Uploading & Parsing Resumes](#uploading--parsing-resumes)
9. [Job Profile Management](#job-profile-management)
10. [Matching Resumes to Jobs](#matching-resumes-to-jobs)
11. [RAG Query Endpoint](#rag-query-endpoint)
12. [License](#license)

---

## Features

* **PDF Resume Parsing**: Extract text, entities (person, organization, dates, etc.), skills, education, experience, and contact details from uploaded resumes using PyMuPDF and SpaCy.
* **Semantic Search & Matching**: Build and query a FAISS index of resume embeddings (via Cohere) to find similar documents and match resumes to job profiles.
* **Skill Scoring & Project Grading**: Grade extracted skills and projects, providing actionable improvement tips.
* **User Authentication**: JWT-based auth with user roles (admin, regular users) and OAuth2 password flow.
* **RESTful API**: Endpoints to manage users, resumes, job profiles, and perform matching.
* **RAG (Retrieval-Augmented Generation)**: Use Cohere to evaluate resumes in context of custom queries.
* **CORS & Static Files**: Easy integration with a React frontend or other clients.

## Prerequisites

* Python 3.8+
* Node.js 14+ (if using the optional React admin portal)
* MongoDB instance (local or remote)
* Cohere API key for embeddings and generation

## Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/resume-parser-api.git
   cd resume-parser-api
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Create an `.env` file in the project root and configure the required variables (see [Configuration](#configuration)).

## Configuration

Rename `.env.example` to `.env` and update the following values:

```dotenv
# MongoDB connection string
MONGO_URL=mongodb://localhost:27017

# JWT secret key
SECRET_KEY=your-secure-secret-key

# Admin user password (optional override)
ADMIN_PASSWORD=admin123

# Cohere API key for embeddings & generation
COHERE_API_KEY=your-cohere-api-key
```

## Installing Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> **Note**: If `en_core_web_lg` is not found, the app will automatically download it on startup.

## Running the Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

* Access the interactive Swagger UI at: `http://localhost:8000/docs`
* Access ReDoc at: `http://localhost:8000/redoc`

## API Endpoints

### Authentication

| Endpoint    | Method | Description                                      |
| ----------- | ------ | ------------------------------------------------ |
| `/token`    | POST   | Obtain JWT access token using username/password. |
| `/users`    | POST   | Create a new user account.                       |
| `/users/me` | GET    | Retrieve current authenticated user info.        |

### Resume Operations

| Endpoint                     | Method | Description                                   |
| ---------------------------- | ------ | --------------------------------------------- |
| `/resumes/upload`            | POST   | Upload PDF resume, parse, store, and index.   |
| `/resumes`                   | GET    | List all uploaded resumes (metadata only).    |
| `/resumes/{resume_id}`       | GET    | Retrieve full parsed resume details.          |
| `/resumes/{resume_id}/score` | POST   | Score a resume and get skill/project metrics. |
| `/resumes/{resume_id}`       | DELETE | Delete a resume and its file.                 |

### Job Profile Operations

| Endpoint                 | Method | Description                                              |
| ------------------------ | ------ | -------------------------------------------------------- |
| `/job-profiles`          | POST   | Create a new job profile with required/preferred skills. |
| `/job-profiles`          | GET    | List all job profiles.                                   |
| `/job-profiles/{job_id}` | GET    | Retrieve a specific job profile.                         |
| `/job-profiles/{job_id}` | DELETE | Delete a job profile.                                    |

### Matching & RAG

| Endpoint                 | Method | Description                                                   |
| ------------------------ | ------ | ------------------------------------------------------------- |
| `/match/{job_id}`        | POST   | Match all resumes against a job profile.                      |
| `/rag/query/{resume_id}` | POST   | Retrieval-augmented evaluation of a resume with custom query. |

## Uploading & Parsing Resumes

1. Send a `POST` to `/resumes/upload` with form-data:

   * **file**: PDF resume
   * **Authorization**: Bearer token

2. The response includes the parsed fields (name, skills, education, experience) and FAISS indexing.

## Job Profile Management

1. Create a job profile via `POST /job-profiles` with JSON body:

   ```json
   {
     "title": "Backend Engineer",
     "description": "Build and maintain APIs...",
     "required_skills": ["python", "fastapi", "mongodb"],
     "preferred_skills": ["docker", "kubernetes"]
   }
   ```

2. Use `GET /job-profiles` to list and `GET /job-profiles/{id}` to view.

## Matching Resumes to Jobs

Trigger resume-job matching with:

```bash
curl -X POST \
  http://localhost:8000/match/{job_id} \
  -H "Authorization: Bearer <token>"
```

The response returns a sorted list of candidates with similarity scores and skill-match breakdowns.

## RAG Query Endpoint

Use `/rag/query/{resume_id}` to ask free-text questions about a resume. Example:

```json
POST /rag/query/608c1f5b2b3e4b5a6c7d8e9f
{
  "query": "How could this resume be improved for a data science role?"
}
```

You’ll receive a scored evaluation, strengths, areas for improvement, and actionable suggestions.

## License

This project is licensed under the [MIT License](LICENSE).
