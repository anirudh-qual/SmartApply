from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import fastapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import PyPDF2
import io
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Resume MCQ Generator Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")


# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Global storage for quiz results
quiz_results = []

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini API connection
        genai.list_models()
        return {
            "status": "healthy", 
            "service": "Resume MCQ Generator",
            "gemini_api": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Resume MCQ Generator", 
            "error": str(e)
        }

# Pydantic models for structured output
class MCQOption(BaseModel):
    option: str
    is_correct: bool

class MCQuestion(BaseModel):
    question: str
    options: List[MCQOption] = Field(..., min_items=4, max_items=4)
    

class MCQResponse(BaseModel):
    questions: List[MCQuestion] = Field(..., min_items=5, max_items=5)

class ApplicantInfo(BaseModel):
    name: str
    email: str
    position: str
    application_id: Optional[str] = None

class QuizResult(BaseModel):
    question_id: int
    selected_option: int
    is_correct: bool
    time_taken_seconds: float

class QuizSubmission(BaseModel):
    applicant_info: ApplicantInfo
    results: List[QuizResult]

class FinalResponse(BaseModel):
    applicant_info: ApplicantInfo
    quiz_score: float
    total_questions: int
    correct_answers: int
    questions_performance: List[dict]
    resume_skills: List[str]


def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text content from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")


def extract_skills_from_resume(resume_text: str) -> List[str]:
    """
    Extract skills that are typically bolded or in skills section
    This is a simplified version - you may want to enhance with NLP
    """
    # Common skill keywords to look for
    skill_keywords = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
        "machine learning", "deep learning", "nlp", "computer vision",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "fastapi", "django", "flask", "spring boot", "node.js", "express",
        "git", "ci/cd", "jenkins", "github actions", "agile", "scrum",
        "microservices", "rest api", "graphql", "kafka", "rabbitmq"
    ]
    
    resume_lower = resume_text.lower()
    found_skills = []
    
    for skill in skill_keywords:
        if skill in resume_lower:
            found_skills.append(skill)
    
    return found_skills[:10]  # Return top 10 found skills


def generate_mcqs_with_gemini(resume_text: str, skills: List[str]) -> MCQResponse:
    """
    Generate MCQs using Gemini with structured output (guided decoding)
    """
    # Define a flat JSON schema for the response to avoid $defs issues
    QUIZ_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "option": {"type": "string"},
                                    "is_correct": {"type": "boolean"}
                                },
                                "required": ["option", "is_correct"]
                            },
                        },
                    },
                    "required": ["question", "options"]
                },
            }
        },
        "required": ["questions"]
    }
    
    try:
        # Initialize Gemini model with JSON schema for structured output
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
                "response_schema": QUIZ_RESPONSE_SCHEMA
            }
        )
        
        prompt = f"""
You are an expert MCQ creator. Using the following resume text and list of technical skills,
generate exactly 5 multiple-choice questions (MCQs) that assess the candidate's knowledge and understanding.

Each question should:
- Be directly related to the candidate's skills.
- Have exactly 4 options, with exactly one marked as correct (is_correct: true for the correct option, false for others).
- Be moderately challenging, targeting practical knowledge.
- Include a skill and difficulty level.

Return ONLY JSON matching the given response_schema.

Resume:
{resume_text}

Identified Skills:
{', '.join(skills)}
"""

        response = model.generate_content(prompt)
        return MCQResponse.model_validate_json(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")


@app.post("/generate-quiz")
async def generate_quiz(
    request: fastapi.Request,
    name: str = Form(...),
    email: str = Form(...),
    position: str = Form(...),
    resume: UploadFile = File(...)
):
    """
    Endpoint to receive applicant information and resume from client, 
    extract text from PDF, and generate MCQ questions based on skills
    """
    print("RECEIVED REQUEST")
    
    # Validate file type
    if not resume.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate file size (max 10MB)
    resume_bytes = await resume.read()
    if len(resume_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Resume file too large. Maximum size is 10MB")
    
    # Extract text from the uploaded PDF
    resume_text = extract_text_from_pdf(resume_bytes)
    
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from resume")
    
    # Extract skills from resume
    skills = extract_skills_from_resume(resume_text)
    
    if not skills:
        raise HTTPException(status_code=400, detail="Could not identify any technical skills in resume")
    
    # Generate MCQs using Gemini with structured output
    mcq_response = generate_mcqs_with_gemini(resume_text, skills)
    
    # Prepare response
    response = {
        "applicant_info": {
            "name": name,
            "email": email,
            "position": position
        },
        "identified_skills": skills,
        "questions": [q.model_dump() for q in mcq_response.questions]
    }
    return JSONResponse(content=response)


@app.post("/submit-quiz", response_model=FinalResponse)
async def submit_quiz(submission: QuizSubmission):
    """
    Endpoint to receive quiz results and calculate performance metrics
    """
    
    total_questions = len(submission.results)
    correct_answers = sum(1 for result in submission.results if result.is_correct)
    quiz_score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    # Prepare detailed performance breakdown
    questions_performance = []
    for result in submission.results:
        questions_performance.append({
            "question_id": result.question_id,
            "selected_option": result.selected_option,
            "is_correct": result.is_correct,
            "time_taken_seconds": result.time_taken_seconds
        })
    
    # Calculate average time
    avg_time = sum(r.time_taken_seconds for r in submission.results) / total_questions if total_questions > 0 else 0
    
    final_response = FinalResponse(
        applicant_info=submission.applicant_info,
        quiz_score=round(quiz_score, 2),
        total_questions=total_questions,
        correct_answers=correct_answers,
        questions_performance=questions_performance,
        resume_skills = [] #Placeholder, can be filled if needed
    )
    
    # Store quiz result
    quiz_results.append({
        "application_id": submission.applicant_info.application_id,
        "quiz_score": round(quiz_score, 2),
        "submitted_at": "2025-11-08T16:39:47"  # Placeholder, use datetime
    })
    
    return final_response


@app.get("/admin/quiz-results")
async def get_quiz_results():
    """Get all quiz results for admin dashboard"""
    return {"quiz_results": quiz_results}


# Mount static files LAST so API routes take precedenc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcq:app", host="0.0.0.0", port=8080, reload=True)