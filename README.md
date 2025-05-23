Resume Parser and Matching System
A complete system for parsing resumes, extracting key information using NLP, and matching candidates to job profiles.
Features

PDF resume parsing with entity extraction using SpaCy
Resume-job matching using sentence embeddings
User authentication and authorization
REST API for all operations
React admin portal

Setup Instructions
Prerequisites

Python 3.8+
Node.js 14+
MongoDB

Backend Setup

Install Python dependencies:
pip install fastapi uvicorn motor spacy sentence-transformers PyMuPDF python-multipart python-jose[cryptography] bcrypt
python -m spacy download en_core_web_lg


Start MongoDB:
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo
# Or use your local MongoDB installation


Set environment variables (optional):
export MONGO_URL="mongodb://localhost:27017"
export SECRET_KEY="your-secure-secret-key"
export ADMIN_PASSWORD="your-admin-password"


Run the backend:
python app.py



Frontend Setup

Install Node.js dependencies:
cd frontend
npm install


Run the frontend:
npm start



API Documentation

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

Default Admin Account

Username: admin
Password: admin123 (or value of ADMIN_PASSWORD environment variable)

