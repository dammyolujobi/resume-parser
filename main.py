from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from difflib import get_close_matches
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import motor.motor_asyncio
from typing import Tuple
import spacy
import os
import fitz
import json
import uuid
import jwt
import bcrypt
import dateparser
import fitz
import re  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from bson import ObjectId, json_util
import faiss
import numpy as np
import cohere
from models import UserInDB,UserDisplay,Token,TokenData,Skill,ContactInfo,ResumeData,JobProfile,CandidateMatch,UserCreate,RAGQuery,RAGResponse,ResumeHit
# -- RAG config
COHERE_API_KEY = "Psgf5ZaD224nY4PPp29tXwYcNe6zMXXclpQ0Dsoi"

if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY is required")
co = cohere.Client(COHERE_API_KEY)

EMBED_MODEL = "embed-english-v2.0"
EMBED_DIM = 4096
INDEX_FILE = "faiss_index.bin"

# load or create index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatIP(EMBED_DIM)

# in-memory metadata list
metadata: List[dict] = []

async def build_faiss_index():
    """Rebuild both the FAISS index and metadata list from the database"""
    # Clear both index and metadata
    global index, metadata
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    index = faiss.IndexFlatIP(EMBED_DIM)
    metadata = []

    resumes = await db.resumes.find().to_list(1000)
    for r in resumes:
        text = r["raw_text"]
        # embed, normalize, add
        resp = co.embed(model=EMBED_MODEL, texts=[text])
        emb = np.array(resp.embeddings[0], dtype="float32")
        emb /= np.linalg.norm(emb)
        index.add(np.expand_dims(emb, 0))
        metadata.append({
            "id": str(r["_id"]),
            "candidate_name": r["candidate_name"],
            "text": text
        })

    # Save both to disk
    faiss.write_index(index, INDEX_FILE)
    save_metadata()
    logger.info(f"Built index with {index.ntotal} vectors and metadata with {len(metadata)} items")

# Update the startup function to load both index and metadata
async def load_index_and_metadata():
    """Load both the FAISS index and metadata list from disk"""
    global index, metadata
    
    # Load the FAISS index
    if os.path.exists(INDEX_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            logger.info(f"Loaded index with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            index = faiss.IndexFlatIP(EMBED_DIM)
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        logger.info("Created new empty FAISS index")
    
    # Load the metadata
    await load_metadata()
    
    # Verify consistency
    if index.ntotal != len(metadata):
        logger.warning(f"Index size ({index.ntotal}) doesn't match metadata size ({len(metadata)})")
        if index.ntotal > 0 and len(metadata) > 0:
            logger.warning("Will rebuild index and metadata for consistency")
            await build_faiss_index()
# Startup event to create admin user if not exists
@asynccontextmanager
async def startup_event(app:FastAPI):
    # Create admin user if not exists
    admin = await db.users.find_one({"username": "admin"})
    logger.info(f"Admin: {admin}")
    if not admin:
        admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        hashed_password = get_password_hash(admin_password)
        await db.users.insert_one({
            "username": "admin",
            "email": "admin@example.com",
            "hashed_password": hashed_password,
            "is_admin": True,
            "created_at": datetime.now()
        })
        logger.info("Admin user created")
        await build_faiss_index()
    try:
        yield
    finally:
        print("Shuttting down")
# Initialize FastAPI
app = FastAPI(title="Resume Parser API", description="API for parsing and matching resumes to job profiles", lifespan=startup_event)

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
MONGO_URL = os.environ.get("MONGO_URL", "mongodb+srv://daolabmovies:Jesusislord@cluster0.2mjk3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
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


# Authentication functions
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password)

def get_password_hash(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

async def get_user(username: str):
    user = await db.users.find_one({"username": username})
    if user:

        return UserInDB(
            _id= user.get("_id"),
            hashed_password = user.get("hashed_password"),
            username=user.get("username"),
            email=user.get("email"),
            created_at = user.get("created_at"),
            is_admin=user.get("is_admin")
            )
    return None

async def authenticate_user(username: str, password: str):
    user = await get_user(username)
    logger.info(f"User: {user}")
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
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    # Pylance should infer username is str here due to the check
    user = await get_user(username=payload.get("sub", ""))#ignore
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    return current_user

@app.post("/rag/query/{resume_id}", response_model=RAGResponse)
async def rag_query(
    resume_id: str,
    req: RAGQuery,
    user=Depends(get_current_active_user),
):
    # 1) Fetch and parse the resume document
    raw_doc = await db.resumes.find_one({"_id": ObjectId(resume_id)})
    if not raw_doc:
        raise HTTPException(status_code=404, detail="Resume not found")
    resume = ResumeData(**raw_doc)

    # 2) Use the full raw_text as context
    content = resume.raw_text
    if not content:
        raise HTTPException(status_code=404, detail="No resume content available for review")

    # 3) Construct prompt
    prompt = f"""
You are a professional resume reviewer. Your task is to evaluate the provided resume content in the context of the user’s query.

Question: {req.query}

Candidate: {resume.candidate_name}

Resume content:
{content}

Please:
1. **Score the resume** (0–100) based on formatting, content quality, clarity, and relevance to backend engineering.
2. **List strengths**.
3. **List areas for improvement**.
4. **Give actionable suggestions**.

Structure your answer with clear headings and bullet points.
"""

    # 4) Call the language model
    gen = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=500,
        temperature=0.3,
        k=0,
        p=0.75,
        stop_sequences=["--"],
    )

    # 5) Build the response
    hit = ResumeHit(
        candidate_name=resume.candidate_name,
        snippet=content
    )
    return RAGResponse(
        results=[hit],
        answer=gen.generations[0].text.strip(),
    )

# Resume parsing function
async def parse_resume(file_path):
    try:
        # Extract text from PDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            page: fitz.Page
            text += page.get_text()  # type: ignore[attr-defined]

        
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
        raw_skills = extract_skills(text)
        skills = [Skill(name=name,score=score) for name,score in raw_skills]
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
        contact = extract_contact_info(text)
        # Create resume data
        resume_data = ResumeData(
            candidate_name=candidate_name,
            contact_info=ContactInfo(**contact),
            skills=skills,
            education=education,
            experience=experience,
            raw_text=text,
            embedding=embedding,
            file_path=file_path,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        return resume_data
    
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")

def retrieve(query: str, top_k: int = 3):
    if index.ntotal == 0:
        # No documents in the index
        return []
    
    # Check if metadata is empty or doesn't match index size
    if not metadata or len(metadata) != index.ntotal:
        logger.warning(f"Metadata list ({len(metadata)}) and index size ({index.ntotal}) are out of sync")
        # Option 1: Rebuild index (could be expensive)
        # await build_faiss_index()
        # Option 2: Return empty results to avoid crash
        return []
        
    # Embed and normalize the query
    qemb = np.array(co.embed(model=EMBED_MODEL, texts=[query]).embeddings[0], dtype="float32")
    qemb /= np.linalg.norm(qemb)
    logger.info(f"Index dimension: {index.d}")
    logger.info(f"Query embedding dimension: {qemb.shape[0]}")

    # Search the index
    D, I = index.search(np.expand_dims(qemb, 0), min(top_k, index.ntotal))
    results = []
    
    for score, idx in zip(D[0], I[0]):
        # Safety check to prevent index out of range errors
        if idx >= 0 and idx < len(metadata):
            doc = metadata[idx]
            snippet = doc["text"]
            results.append({
                "id": doc["id"],
                "candidate_name": doc["candidate_name"],
                "snippet": snippet,
                "score": float(score)
            })
        else:
            logger.error(f"Index {idx} out of bounds for metadata list of size {len(metadata)}")
    
    return results

def save_metadata():
    """Save metadata to disk"""
    try:
        with open("metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        logger.info(f"Saved metadata with {len(metadata)} items")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")

async def load_metadata():
    """Load metadata from disk"""
    global metadata
    try:
        if os.path.exists("metadata.json"):
            with open("metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} items")
        else:
            # Rebuild metadata from database
            metadata = []
            resumes = await db.resumes.find().to_list(1000)
            for r in resumes:
                metadata.append({
                    "id": str(r["_id"]),
                    "candidate_name": r["candidate_name"],
                    "text": r["raw_text"]
                })
            logger.info(f"Rebuilt metadata with {len(metadata)} items")
            save_metadata()
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        metadata = []

# Update the add_to_index function to properly save metadata
def add_to_index(doc_id: str, raw_text: str, candidate_name: str):
    """Add a document to the FAISS index and metadata list"""
    # 1) Embed
    resp = co.embed(model=EMBED_MODEL, texts=[raw_text])
    emb = np.array(resp.embeddings[0], dtype="float32")
    emb /= np.linalg.norm(emb)

    # 2) Add to FAISS
    index.add(np.expand_dims(emb, 0))
    
    # 3) Append metadata
    metadata.append({
        "id": doc_id,
        "candidate_name": candidate_name,
        "text": raw_text
    })
    
    # 4) Save both to disk
    try:
        faiss.write_index(index, INDEX_FILE)
        save_metadata()
    except Exception as e:
        logger.error(f"Error saving index or metadata: {str(e)}")


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
    
    found = {}
    text_lower = text.lower()
    # Direct exact matches
    for skill in common_skills:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found[skill] = 1.0

    # Fuzzy matches for variations
    tokens = set(re.findall(r"\b[\w.+#]+\b", text_lower))
    for token in tokens:
        matches = get_close_matches(token, common_skills, cutoff=0.85)
        for m in matches:
            # Avoid overriding exact match score
            if m not in found:
                found[m] = 0.8

    # Convert to list of (skill, score)
    return [(skill, score) for skill, score in found.items()]

def extract_education(text: str, nlp_model) -> list:
    edu_list = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    edu_patterns = [r"Bachelor.+", r"Master.+", r"PhD.+", r"Diploma.+"]
    for i, line in enumerate(lines):
        for pat in edu_patterns:
            if re.search(pat, line, re.IGNORECASE):
                inst = None
                # next line may have institution name
                if i+1 < len(lines) and any(word in lines[i+1].lower() for word in ['university', 'college', 'institute', 'school']):
                    inst = lines[i+1]
                else:
                    # try to pull from current line
                    inst = line

                # extract dates around this block
                date_matches = re.findall(r"(\b\w+ \d{4}\b)", line + ' ' + (lines[i+1] if i+1<len(lines) else ''))
                dates = [dateparser.parse(d) for d in date_matches]
                start, end = None, None
                if dates:
                    dates_sorted = sorted([d for d in dates if d])
                    start = dates_sorted[0].date().isoformat()
                    if len(dates_sorted) > 1:
                        end = dates_sorted[-1].date().isoformat()

                edu = {
                    'institution': inst,
                    'degree': re.search(r"(Bachelor.+|Master.+|PhD.+|Diploma.+)", line, re.IGNORECASE).group(0),
                    'field_of_study': None,
                    'start_date': start,
                    'end_date': end
                }
                edu_list.append(edu)
    return edu_list

def extract_experience(text: str, nlp_model) -> list:
    exp_list = []
    # split at bullet points or lines with date ranges
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        # match date range like Jan 2020 - Feb 2021
        dr = re.search(r"(\b\w+ \d{4}\b)\s*[-–—]\s*(\b\w+ \d{4}\b)", line)
        if dr:
            title_company = re.split(r"\d{4}", line)[0].strip(' -–—')
            parts = [p.strip() for p in title_company.split(' at ')]
            title = parts[0]
            start = dateparser.parse(dr.group(1)).date().isoformat()
            end = dateparser.parse(dr.group(2)).date().isoformat()
            exp_list.append({
                'company': title_company,
                'title': title,
                'start_date': start,
                'end_date': end,
                'description': None
            })
    return exp_list

def extract_contact_info(text: str):
    contact = {}
    # Email
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    if emails:
        contact['email'] = emails[0]

    # Phone (international formats)
    phones = re.findall(r"\+?\d[\d\s\-()]{7,}\d", text)
    if phones:
        contact['phone'] = phones[0]

    # LinkedIn URL
    linkedin = re.search(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+", text)
    if linkedin:
        contact['linkedin'] = linkedin.group(0)

    # Personal website
    websites = re.findall(r"https?://[A-Za-z0-9\-_.]+\.[A-Za-z]{2,}(/[A-Za-z0-9\-_.]*)*", text)
    for site in websites:
        if 'linkedin.com' not in site:
            contact['website'] = site
            break

    return contact

def grade_projects(projects: List[Dict]) -> Tuple[float, List[str]]:
    """Return (project_percentage, improvement_tips)."""
    tips = []
    points = 0

    # # of projects
    n = min(len(projects), 5)
    points += n
    if len(projects) < 3:
        tips.append("Consider adding more projects (aim for 3–5).")

    # Tech diversity
    categories = set()
    for p in projects:
        desc = p.get("description","").lower()
        for cat in ("backend","frontend","ml","mobile","data","automation"):
            if cat in desc:
                categories.add(cat)
    tech_points = min(len(categories), 5)
    points += tech_points
    if tech_points < 3:
        tips.append("Expose a wider variety of technologies (e.g., add a mobile or ML project).")

    # Impact
    for p in projects:
        if re.search(r"\b\d+ (users|downloads|%|growth)\b", p.get("description","")):
            points += 4
        elif len(p.get("description","").split("\n")) >= 2:
            points += 2
        else:
            tips.append(f"Project '{p.get('name')}' could benefit from more measurable outcomes.")

    # Description detail
    bullets = sum(len(p.get("description","").split("\n")) for p in projects)
    text_points = min(bullets // 2, 4)
    points += text_points
    if text_points < 4:
        tips.append("Use at least 2–4 bullet points per project to highlight responsibilities and achievements.")

    # Best practices
    for p in projects:
        yes = 0
        for practice in ("test","ci/cd","docker","kubernetes","coverage"):
            if practice in p.get("description","").lower():
                yes += 1
        points += min(yes,5)
        if yes < 2:
            tips.append(f"Add details on CI/CD, testing or containerization in '{p.get('name')}'.")

    # Live demo / repo link
    for p in projects:
        if p.get("url"):
            points += 2
        else:
            tips.append(f"Provide a GitHub link or live demo for '{p.get('name')}'.")

    # Cap total at 25
    points = min(points, 25)
    percentage = round(points / 25 * 100, 2)
    return percentage, tips


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
    percent_score = round(final_score * 100, 2)  # e.g. 83.27

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
async def create_user(user: UserCreate):
    # Check if user already exists
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash the password
    hashed_password = get_password_hash(user.password)
    
    # Create user document
    user_dict = user.model_dump(exclude={"password"})
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
    user_dict = current_user.model_dump()
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
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Parse resume
    resume_data = await parse_resume(file_path)
    
    # Save to database
    resume_dict = resume_data.model_dump()
    result = await db.resumes.insert_one(resume_dict)
    add_to_index(str(result.inserted_id), resume_data.raw_text, resume_data.candidate_name)

    
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
@app.post("/resumes/{resume_id}/score", response_model=Dict)
async def score_resume(resume_id: str, current_user=Depends(get_current_active_user)):
    # 1. Fetch resume
    resume = await db.resumes.find_one({"_id": ObjectId(resume_id)})
    if not resume:
        raise HTTPException(404, "Resume not found")
    data = ResumeData(**resume)
    # 2. Compute job‐match percent (use a default “benchmark” job or accept job_id param)
    #    Here we skip job matching and just score the resume itself
    skill_coverage = sum(s.score for s in data.skills) / max(len(data.skills),1)
    skill_pct = round(skill_coverage * 100, 2)
    # 3. Grade projects
    proj_pct, proj_tips = grade_projects(data.experience)  
    # 4. Overall “resume health” could be avg of sections
    overall_pct = round((skill_pct + proj_pct) / 2, 2)
    return {
        "resume_id": resume_id,
        "overall_percentage": overall_pct,
        "skill_percentage": skill_pct,
        "project_percentage": proj_pct,
        "project_tips": proj_tips
    }

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
        proj_list = resume_data.experience  # or a new field resume_data.projects
        proj_pct, proj_tips = grade_projects(proj_list)

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