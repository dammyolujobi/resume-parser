from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import motor.motor_asyncio
import spacy
import os
import json
import uuid
import jwt
import bcrypt
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from bson import ObjectId, json_util

# Initialize FastAPI
app = FastAPI(title="Resume Parser API", description="API for parsing and matching resumes to job profiles")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SpaCy NER model
try:
    nlp = spacy.load("en_core_web_lg")
except:
    # If model not found, download it
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

# Initialize Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.resume_parser

# JWT Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-for-jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# File upload configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create JSON encoder that can handle ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

# Pydantic models
class User(BaseModel):
    username: str
    email: str
    password: str
    is_admin: bool = False

class UserInDB(User):
    hashed_password: str

class UserDisplay(BaseModel):
    id: str
    username: str
    email: str
    is_admin: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class Skill(BaseModel):
    name: str
    score: Optional[float] = 0

class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None

class ResumeData(BaseModel):
    id: Optional[str] = None
    candidate_name: str
    contact_info: ContactInfo
    skills: List[Skill] = []
    education: List[Education] = []
    experience: List[Experience] = []
    raw_text: str
    embedding: Optional[List[float]] = None
    file_path: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class JobProfile(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    education_requirements: List[str] = []
    experience_years: Optional[int] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class CandidateMatch(BaseModel):
    resume_id: str
    candidate_name: str
    match_score: float
    skill_matches: Dict[str, float]

# Authentication functions
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password)

def get_password_hash(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

async def get_user(username: str):
    user = await db.users.find_one({"username": username})
    if user:
        return UserInDB(**user)
    return None

async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    return current_user

# Resume parsing function
async def parse_resume(file_path):
    try:
        # Extract text from PDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Process with SpaCy for NER
        doc = nlp(text)
        
        # Extract entities
        entities = {
            "PERSON": [],
            "ORG": [],
            "DATE": [],
            "GPE": [],  # Geo-Political Entity
            "SKILL": []  # We'll need to enhance this later as SpaCy doesn't have a SKILL entity
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # Extract skills (requires custom logic beyond basic NER)
        skills = extract_skills(text)
        
        # Extract education
        education = extract_education(text, entities)
        
        # Extract experience
        experience = extract_experience(text, entities)
        
        # Extract contact information
        contact_info = extract_contact_info(text)
        
        # Get candidate name
        candidate_name = get_candidate_name(entities["PERSON"])
        
        # Create embedding for the resume
        embedding = sentence_model.encode(text).tolist()
        
        # Create resume data
        resume_data = ResumeData(
            candidate_name=candidate_name,
            contact_info=contact_info,
            skills=[Skill(name=skill) for skill in skills],
            education=education,
            experience=experience,
            raw_text=text,
            embedding=embedding,
            file_path=file_path,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return resume_data
    
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")

# Helper functions for entity extraction
def extract_skills(text):
    # This is a simplified approach - in a production system, you'd want a more robust skill extraction
    # You could use a pre-defined list of skills or a custom ML model
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue", "html", "css",
        "node.js", "express", "django", "flask", "fastapi", "aws", "azure", "gcp", "docker",
        "kubernetes", "terraform", "ci/cd", "git", "agile", "scrum", "product management",
        "project management", "machine learning", "data science", "artificial intelligence",
        "nlp", "computer vision", "sql", "nosql", "mongodb", "postgresql", "mysql",
        "leadership", "teamwork", "communication", "problem solving", "critical thinking"
    ]
    
    skills = []
    text_lower = text.lower()
    
    for skill in common_skills:
        if skill in text_lower:
            skills.append(skill)
    
    return skills

def extract_education(text, entities):
    education_list = []
    
    # Look for common education keywords
    education_keywords = ["bachelor", "master", "phd", "degree", "university", "college"]
    
    # Simple approach: look for organizations that might be educational institutions
    # In a production system, you'd want a more sophisticated approach
    for org in entities["ORG"]:
        if any(keyword in org.lower() for keyword in ["university", "college", "institute", "school"]):
            education_list.append(Education(institution=org))
    
    return education_list

def extract_experience(text, entities):
    experience_list = []
    
    # Simple approach: assume organizations are companies
    # In a production system, you'd want to match with job titles, dates, etc.
    for org in entities["ORG"]:
        if not any(keyword in org.lower() for keyword in ["university", "college", "institute", "school"]):
            experience_list.append(Experience(company=org, title="Unknown Position"))
    
    return experience_list

def extract_contact_info(text):
    # Extract email
    email = None
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    import re
    emails = re.findall(email_pattern, text)
    if emails:
        email = emails[0]
    
    # Extract phone (simplified pattern)
    phone = None
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        phone = phones[0]
    
    return ContactInfo(email=email, phone=phone)

def get_candidate_name(person_entities):
    # Simplified approach: assume the first PERSON entity is the candidate
    if person_entities:
        return person_entities[0]
    return "Unknown Candidate"

# Job profile matching function
async def match_resume_to_job(resume_data, job_profile):
    # Calculate overall similarity based on text embeddings
    resume_embedding = np.array(resume_data.embedding).reshape(1, -1)
    job_embedding = np.array(job_profile.embedding).reshape(1, -1)
    overall_similarity = float(cosine_similarity(resume_embedding, job_embedding)[0][0])
    
    # Calculate skill matches
    skill_matches = {}
    resume_skills = [skill.name.lower() for skill in resume_data.skills]
    
    # Check required skills
    for skill in job_profile.required_skills:
        skill_lower = skill.lower()
        if skill_lower in resume_skills:
            skill_matches[skill] = 1.0
        else:
            # Check if similar skills exist
            best_match = 0
            for resume_skill in resume_skills:
                # Calculate similarity between job skill and resume skill
                skill_similarity = float(cosine_similarity(
                    sentence_model.encode([skill_lower]).reshape(1, -1),
                    sentence_model.encode([resume_skill]).reshape(1, -1)
                )[0][0])
                
                if skill_similarity > 0.7:  # Threshold for similarity
                    best_match = max(best_match, skill_similarity)
            
            skill_matches[skill] = best_match
    
    # Check preferred skills (with lower weight)
    for skill in job_profile.preferred_skills:
        skill_lower = skill.lower()
        if skill_lower in resume_skills:
            skill_matches[skill] = 0.7  # Lower weight for preferred skills
        else:
            # Check if similar skills exist
            best_match = 0
            for resume_skill in resume_skills:
                # Calculate similarity between job skill and resume skill
                skill_similarity = float(cosine_similarity(
                    sentence_model.encode([skill_lower]).reshape(1, -1),
                    sentence_model.encode([resume_skill]).reshape(1, -1)
                )[0][0])
                
                if skill_similarity > 0.7:  # Threshold for similarity
                    best_match = max(best_match, skill_similarity)
            
            skill_matches[skill] = best_match * 0.7  # Lower weight for preferred skills
    
    # Calculate weighted score
    skill_score = sum(skill_matches.values()) / max(len(skill_matches), 1)
    final_score = 0.6 * overall_similarity + 0.4 * skill_score
    
    return CandidateMatch(
        resume_id=str(resume_data.id),
        candidate_name=resume_data.candidate_name,
        match_score=final_score,
        skill_matches=skill_matches
    )

# API Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=UserDisplay)
async def create_user(user: User):
    # Check if user already exists
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash the password
    hashed_password = get_password_hash(user.password)
    
    # Create user document
    user_dict = user.dict()
    user_dict.pop("password")
    user_dict["hashed_password"] = hashed_password
    user_dict["created_at"] = datetime.utcnow()
    
    # Insert into database
    result = await db.users.insert_one(user_dict)
    
    # Return user data
    created_user = await db.users.find_one({"_id": result.inserted_id})
    created_user["id"] = str(created_user.pop("_id"))
    return UserDisplay(**created_user)

@app.get("/users/me", response_model=UserDisplay)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    user_dict = current_user.dict()
    user_dict.pop("hashed_password")
    user_dict["id"] = str(user_dict.pop("_id", "unknown"))
    return UserDisplay(**user_dict)

@app.post("/resumes/upload")
async def upload_resume(
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Parse resume
    resume_data = await parse_resume(file_path)
    
    # Save to database
    resume_dict = resume_data.dict()
    result = await db.resumes.insert_one(resume_dict)
    
    # Update with ID
    resume_data.id = str(result.inserted_id)
    
    return {
        "id": resume_data.id,
        "filename": file.filename,
        "candidate_name": resume_data.candidate_name,
        "message": "Resume uploaded and parsed successfully"
    }

@app.get("/resumes", response_model=List[Dict])
async def get_resumes(current_user: UserInDB = Depends(get_current_active_user)):
    resumes = await db.resumes.find().to_list(1000)
    
    # Process for response
    result = []
    for resume in resumes:
        resume["id"] = str(resume.pop("_id"))
        # Don't send embedding in the list view to reduce payload size
        if "embedding" in resume:
            del resume["embedding"]
        result.append(resume)
    
    return result

@app.get("/resumes/{resume_id}")
async def get_resume(
    resume_id: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        resume = await db.resumes.find_one({"_id": ObjectId(resume_id)})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Convert ObjectId to string
        resume["id"] = str(resume.pop("_id"))
        
        return json.loads(json_util.dumps(resume))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving resume: {str(e)}")

@app.post("/job-profiles")
async def create_job_profile(
    job_profile: JobProfile,
    current_user: UserInDB = Depends(get_current_active_user)
):
    # Generate embedding for job description
    job_text = f"{job_profile.title} {job_profile.description} {' '.join(job_profile.required_skills)}"
    embedding = sentence_model.encode(job_text).tolist()
    
    # Add embedding and timestamps
    job_dict = job_profile.dict()
    job_dict["embedding"] = embedding
    job_dict["created_at"] = datetime.now()
    job_dict["updated_at"] = datetime.now()
    
    # Save to database
    result = await db.job_profiles.insert_one(job_dict)
    
    # Return with ID
    job_dict["id"] = str(result.inserted_id)
    del job_dict["embedding"]  # Don't return the embedding
    
    return job_dict

@app.get("/job-profiles", response_model=List[Dict])
async def get_job_profiles(current_user: UserInDB = Depends(get_current_active_user)):
    job_profiles = await db.job_profiles.find().to_list(1000)
    
    # Process for response
    result = []
    for profile in job_profiles:
        profile["id"] = str(profile.pop("_id"))
        # Don't send embedding in the list view
        if "embedding" in profile:
            del profile["embedding"]
        result.append(profile)
    
    return result

@app.get("/job-profiles/{job_id}")
async def get_job_profile(
    job_id: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        job = await db.job_profiles.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job profile not found")
        
        # Convert ObjectId to string
        job["id"] = str(job.pop("_id"))
        
        # Don't return the embedding
        if "embedding" in job:
            del job["embedding"]
        
        return job
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job profile: {str(e)}")

@app.post("/match/{job_id}")
async def match_resumes_to_job(
    job_id: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Get job profile
        job = await db.job_profiles.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job profile not found")
        
        job_profile = JobProfile(**job)
        
        # Get all resumes
        resumes = await db.resumes.find().to_list(1000)
        
        # Match each resume
        matches = []
        for resume in resumes:
            resume_data = ResumeData(**resume)
            match = await match_resume_to_job(resume_data, job_profile)
            matches.append(match.model_dump())
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {"matches": matches}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching resumes: {str(e)}")

@app.delete("/resumes/{resume_id}")
async def delete_resume(
    resume_id: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Check if resume exists
        resume = await db.resumes.find_one({"_id": ObjectId(resume_id)})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Delete file if exists
        if "file_path" in resume and os.path.exists(resume["file_path"]):
            os.remove(resume["file_path"])
        
        # Delete from database
        await db.resumes.delete_one({"_id": ObjectId(resume_id)})
        
        return {"message": "Resume deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting resume: {str(e)}")

@app.delete("/job-profiles/{job_id}")
async def delete_job_profile(
    job_id: str,
    current_user: UserInDB = Depends(get_current_active_user)
):
    try:
        # Check if job profile exists
        job = await db.job_profiles.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job profile not found")
        
        # Delete from database
        await db.job_profiles.delete_one({"_id": ObjectId(job_id)})
        
        return {"message": "Job profile deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job profile: {str(e)}")

# Startup event to create admin user if not exists
@asynccontextmanager
async def startup_event(app:FastAPI):
    # Create admin user if not exists
    admin = await db.users.find_one({"username": "admin"})
    if not admin:
        admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        hashed_password = get_password_hash(admin_password)
        await db.users.insert_one({
            "username": "admin",
            "email": "admin@example.com",
            "hashed_password": hashed_password,
            "is_admin": True,
            "created_at": datetime.utcnow()
        })
        logger.info("Admin user created")

# Serve static files (useful for development)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# README for setup and usage
"""
# Resume Parser and Matching System

A complete system for parsing resumes, extracting key information using NLP, and matching candidates to job profiles.

## Features
- PDF resume parsing with entity extraction using SpaCy
- Resume-job matching using sentence embeddings
- User authentication and authorization
- REST API for all operations
- React admin portal

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- MongoDB

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn pymongo motor spacy sentence-transformers PyMuPDF python-multipart python-jose[cryptography] passlib bcrypt
   python -m spacy download en_core_web_lg
   ```

2. Start MongoDB:
   ```bash
   # Using Docker
   docker run -d -p 27017:27017 --name mongodb mongo
   # Or use your local MongoDB installation
   ```

3. Set environment variables (optional):
   ```bash
   export MONGO_URL="mongodb://localhost:27017"
   export SECRET_KEY="your-secure-secret-key"
   export ADMIN_PASSWORD="your-admin-password"
   ```

4. Run the backend:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Run the frontend:
   ```bash
   npm start
   ```

## API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Default Admin Account
- Username: admin
- Password: admin123 (or value of ADMIN_PASSWORD env var)
"""