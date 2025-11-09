from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
import requests
import io
import os
from datetime import datetime
import google.generativeai as genai
import json
from dotenv import load_dotenv
import base64
import librosa
import numpy as np
import tempfile
import time
load_dotenv()


app = FastAPI(title="Job Application API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Pydantic Models
class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "pNInz6obpgDQGcFmaJgB"
    stability: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.7

class JobApplicationResponse(BaseModel):
    message: str
    application_id: str
    submitted_at: str

class InterviewQuestion(BaseModel):
    question_id: int
    question: str
    category: Optional[str] = None

class GenerateQuestionsRequest(BaseModel):
    position: str
    num_questions: int
    difficulty: Optional[str] = "medium"  # easy, medium, hard
    categories: Optional[List[str]] = None  # e.g., ["technical", "behavioral", "situational"]

class QuestionAnswer(BaseModel):
    question_id: int
    question: str
    answer: str

class InterviewSubmission(BaseModel):
    application_id: str
    position: str
    answers: List[QuestionAnswer]

class EvaluationResponse(BaseModel):
    application_id: str
    overall_score: float
    detailed_scores: List[Dict]
    strengths: List[str]
    areas_for_improvement: List[str]
    recommendation: str
    summary: str

# In-memory storage (replace with database in production)
job_applications = []
interview_sessions = {}  # Store interview questions and answers

@app.get("/")
async def root():
    return {
        "message": "Job Application API",
        "endpoints": {
            "POST /api/job-application": "Submit job application with resume",
            "POST /api/text-to-speech": "Convert text to speech audio",
            "GET /api/applications": "Get all applications (admin)"
        }
    }

@app.post("/api/job-application", response_model=JobApplicationResponse)
async def submit_job_application(
    full_name: str = Form(...),
    email: EmailStr = Form(...),
    phone: str = Form(...),
    position: str = Form(...),
    cover_letter: Optional[str] = Form(None),
    resume: UploadFile = File(...)
):
    """Submit job application"""
    try:
        allowed_extensions = ['.pdf', '.doc', '.docx']
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        resume_content = await resume.read()
        application_id = f"APP-{len(job_applications) + 1:05d}"
        submission_time = datetime.now().isoformat()
        
        # Save resume
        resume_dir = "/tmp/resumes"
        os.makedirs(resume_dir, exist_ok=True)
        
        safe_filename = f"{application_id}_{resume.filename}"
        resume_path = os.path.join(resume_dir, safe_filename)
        
        with open(resume_path, "wb") as f:
            f.write(resume_content)
        
        application_data = {
            "application_id": application_id,
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "position": position,
            "cover_letter": cover_letter,
            "resume_filename": resume.filename,
            "resume_path": resume_path,
            "resume_size": len(resume_content),
            "submitted_at": submission_time
        }
        
        job_applications.append(application_data)
        
        return JobApplicationResponse(
            message="Application submitted successfully",
            application_id=application_id,
            submitted_at=submission_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech using ElevenLabs API
    Returns audio file stream
    """
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}"
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": request.text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": request.stability,
                "similarity_boost": request.similarity_boost
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            # Return audio as streaming response
            audio_stream = io.BytesIO(response.content)
            return StreamingResponse(
                audio_stream,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs API error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/applications")
async def get_applications():
    """
    Get all job applications (Admin endpoint - add authentication in production)
    """
    return {
        "total": len(job_applications),
        "applications": job_applications
    }

@app.get("/api/applications/{application_id}")
async def get_application(application_id: str):
    """
    Get specific application by ID
    """
    application = next(
        (app for app in job_applications if app["application_id"] == application_id),
        None
    )
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    return application

@app.get("/api/applications/{application_id}/resume")
async def download_resume(application_id: str):
    """
    Download the resume file for a specific application
    """
    application = next(
        (app for app in job_applications if app["application_id"] == application_id),
        None
    )
    
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    resume_path = application.get("resume_path")
    
    if not resume_path or not os.path.exists(resume_path):
        raise HTTPException(status_code=404, detail="Resume file not found")
    
    # Determine media type based on file extension
    file_extension = os.path.splitext(resume_path)[1].lower()
    media_types = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    media_type = media_types.get(file_extension, 'application/octet-stream')
    
    # Read and return the file
    with open(resume_path, "rb") as f:
        resume_content = f.read()
    
    return StreamingResponse(
        io.BytesIO(resume_content),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={application['resume_filename']}"
        }
    )

@app.get("/api/resumes")
async def list_all_resumes():
    """
    List all resume files with download links
    """
    resumes = []
    
    for app in job_applications:
        resume_path = app.get("resume_path")
        file_exists = resume_path and os.path.exists(resume_path)
        
        resumes.append({
            "application_id": app["application_id"],
            "applicant_name": app["full_name"],
            "position": app["position"],
            "filename": app["resume_filename"],
            "file_size": app["resume_size"],
            "file_exists": file_exists,
            "download_url": f"/api/applications/{app['application_id']}/resume" if file_exists else None,
            "submitted_at": app["submitted_at"]
        })
    
    return {
        "total": len(resumes),
        "resumes": resumes
    }

@app.post("/api/interview/generate-questions")
async def generate_interview_questions(request: GenerateQuestionsRequest):
    """
    Generate interview questions using Gemini API based on position and requirements
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Build prompt for question generation
        categories_str = ", ".join(request.categories) if request.categories else "technical, behavioral, and situational"
        
        prompt = f"""Generate exactly {request.num_questions} interview questions for a {request.position} position.

Difficulty level: {request.difficulty}
Question categories: {categories_str}

Requirements:
1. Generate diverse questions covering different aspects of the role
2. Include a mix of: {categories_str}
3. Questions should be clear, professional, and relevant to {request.position}
4. Difficulty should be {request.difficulty}

Return the response in the following JSON format ONLY (no markdown, no extra text):
{{
    "questions": [
        {{
            "question_id": 1,
            "question": "Question text here",
            "category": "technical/behavioral/situational"
        }}
    ]
}}
"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        questions_data = json.loads(response_text)
        
        # Create session ID
        session_id = f"SESSION-{len(interview_sessions) + 1:05d}"
        
        # Store questions in session
        interview_sessions[session_id] = {
            "position": request.position,
            "difficulty": request.difficulty,
            "questions": questions_data["questions"],
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "session_id": session_id,
            "position": request.position,
            "questions": questions_data["questions"]
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/api/interview/evaluate", response_model=EvaluationResponse)
async def evaluate_interview_answers(submission: InterviewSubmission):
    """
    Evaluate interview answers using Gemini API
    Returns detailed evaluation with scores and feedback
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build evaluation prompt
        answers_text = "\n\n".join([
            f"Q{ans.question_id}: {ans.question}\nAnswer: {ans.answer}"
            for ans in submission.answers
        ])
        
        prompt = f"""You are an expert technical interviewer evaluating a candidate for a {submission.position} position.

Interview Answers:
{answers_text}

Please provide a comprehensive evaluation in the following JSON format ONLY (no markdown, no extra text):
{{
    "overall_score": 0.0,
    "detailed_scores": [
        {{
            "question_id": 1,
            "score": 0.0,
            "feedback": "Specific feedback for this answer"
        }}
    ],
    "strengths": ["List key strengths demonstrated"],
    "areas_for_improvement": ["List areas that need improvement"],
    "recommendation": "hire/maybe/reject with brief explanation",
    "summary": "Brief overall assessment of the candidate"
}}

Scoring Guidelines:
- overall_score: 0-100 scale
- question scores: 0-10 scale
- Be fair, objective, and constructive
- Consider technical accuracy, communication skills, problem-solving approach
- Provide specific, actionable feedback
"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        evaluation_data = json.loads(response_text)
        
        # Store evaluation results
        if submission.application_id not in interview_sessions:
            interview_sessions[submission.application_id] = {}
        
        interview_sessions[submission.application_id]["evaluation"] = {
            **evaluation_data,
            "evaluated_at": datetime.now().isoformat()
        }
        
        return EvaluationResponse(
            application_id=submission.application_id,
            **evaluation_data
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse evaluation response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating answers: {str(e)}")

@app.get("/api/interview/session/{session_id}")
async def get_interview_session(session_id: str):
    """
    Get interview session details including questions and evaluation if available
    """
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    return interview_sessions[session_id]

@app.get("/api/interview/sessions")
async def get_all_sessions():
    """
    Get all interview sessions (Admin endpoint)
    """
    return {
        "total": len(interview_sessions),
        "sessions": interview_sessions
    }

@app.get("/api/admin/storage")
async def get_all_storage():
    """
    Admin endpoint: Get complete view of all in-memory storage
    """
    return {
        "applications": {
            "count": len(job_applications),
            "data": job_applications
        },
        "interview_sessions": {
            "count": len(interview_sessions),
            "data": interview_sessions
        },
        "summary": {
            "total_applications": len(job_applications),
            "total_interview_sessions": len(interview_sessions),
            "applications_with_evaluations": sum(
                1 for session in interview_sessions.values() 
                if isinstance(session, dict) and "evaluation" in session
            )
        }
    }

@app.delete("/api/admin/storage/reset")
async def reset_storage():
    """
    Admin endpoint: Clear all in-memory storage (useful for testing)
    """
    global job_applications, interview_sessions
    
    old_app_count = len(job_applications)
    old_session_count = len(interview_sessions)
    
    job_applications.clear()
    interview_sessions.clear()
    
    return {
        "message": "Storage reset successfully",
        "cleared": {
            "applications": old_app_count,
            "sessions": old_session_count
        }
    }

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, EmailStr
# from typing import Optional, List, Dict
# import requests
# import io
# import os
# from datetime import datetime
# import google.generativeai as genai
# import json
# from dotenv import load_dotenv
# import PyPDF2
# from docx import Document
# import base64
# import librosa
# import numpy as np
# import tempfile

# load_dotenv()

# app = FastAPI(title="Job Application API with Live Interview")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API Configuration
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# Pydantic Models
class JobApplicationResponse(BaseModel):
    message: str
    application_id: str
    submitted_at: str

class StartInterviewRequest(BaseModel):
    application_id: str

class ContinueInterviewRequest(BaseModel):
    session_id: str
    answer_text: str
    audio_blob_base64: Optional[str] = None

# In-memory storage
job_applications = []
interview_sessions = {}

# ==================== HELPER FUNCTIONS ====================

def extract_resume_text(resume_path: str) -> str:
    """Extract text from resume file"""
    try:
        file_ext = os.path.splitext(resume_path)[1].lower()
        return "fjdlksjlasdfjklsdafjlkjdsfkljdsflkjlkdsf"
        # if file_ext == '.pdf':
        #     with open(resume_path, 'rb') as f:
        #         reader = PyPDF2.PdfReader(f)
        #         text = ""
        #         for page in reader.pages:
        #             text += page.extract_text()
        #     return text
        
        # elif file_ext in ['.doc', '.docx']:
        #     doc = Document(resume_path)
        #     return "\n".join([para.text for para in doc.paragraphs])
        
        return ""
    except Exception as e:
        print(f"Error extracting resume text: {e}")
        return ""

def analyze_audio_emotions(audio_base64: str) -> dict:
    """Analyze emotions from audio"""
    try:
        # Decode base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save to temp file - use .webm extension for WebM audio, librosa will handle it
        # If audio format detection fails, we'll try .wav as fallback
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Load with librosa (supports WebM if ffmpeg is available)
            y, sr = librosa.load(temp_path, sr=None)
        except Exception as e:
            # If WebM fails, try saving as WAV and converting
            print(f"WebM load failed, trying alternative: {e}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Try with .wav extension (librosa might still work)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            y, sr = librosa.load(temp_path, sr=None)
        
        # Extract features
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(rms)
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        avg_pitch = np.mean(pitch_values) if pitch_values else 0
        pitch_variance = np.std(pitch_values) if pitch_values else 0
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zcr)
        
        # Clean up
        os.unlink(temp_path)
        
        # Interpret
        energy_level = "high" if avg_energy > 0.05 else "medium" if avg_energy > 0.02 else "low"
        pitch_stability = "stable" if pitch_variance < 50 else "variable"
        speech_rate = "fast" if avg_zcr > 0.1 else "moderate" if avg_zcr > 0.05 else "slow"
        
        if energy_level == "high" and pitch_stability == "stable":
            tone = "confident"
        elif energy_level == "low" and pitch_stability == "variable":
            tone = "nervous"
        elif energy_level == "medium":
            tone = "calm"
        else:
            tone = "neutral"
        
        return {
            "tone": tone,
            "energy": energy_level,
            "pitch_stability": pitch_stability,
            "speech_rate": speech_rate,
            "confidence_score": float(avg_energy * 100)
        }
    except Exception as e:
        print(f"Audio analysis error: {e}")
        return {
            "tone": "neutral",
            "energy": "medium",
            "confidence_score": 50.0
        }

def generate_empathetic_response(audio_data: dict) -> str:
    """Generate empathetic response using Gemini"""
    try:
        tone = audio_data.get('tone', 'neutral')
        energy = audio_data.get('energy', 'medium')
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Generate a brief (1 sentence) empathetic response for an interviewer based on:
- Tone: {tone}
- Energy: {energy}

Be encouraging if nervous, engaging if confident, supportive if low energy.
Return ONLY the empathetic sentence, nothing else."""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Thank you for your response."

def text_to_speech_bytes(text: str, audio_data: dict = None) -> str:
    """Convert text to speech"""
    try:
        # Default settings
        stability = 0.6
        similarity_boost = 0.7
        voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam
        
        if audio_data:
            tone = audio_data.get('tone', 'neutral')
            
            if tone == 'nervous':
                stability = 0.7
                similarity_boost = 0.8
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (warm)
            elif tone == 'confident':
                stability = 0.5
                similarity_boost = 0.7
                voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        
        return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# ==================== API ENDPOINTS ====================

job_applications = []  # Example storage
interview_sessions = {}  # Example session storage

@app.get("/interview")
async def root():
    return {
        "message": "Live Interview System",
        "features": [
            "Resume-based question generation",
            "Audio tone analysis",
            "Gemini AI evaluation",
            "ElevenLabs TTS with emotional voice"
        ]
    }

@app.post("/api/interview/start")
async def start_live_interview(request: StartInterviewRequest):
    """Start interview based on resume"""
    try:
        # Step 1: Fetch application
        app_data = next((a for a in job_applications if a["application_id"] == request.application_id), None)
        if not app_data:
            raise HTTPException(status_code=404, detail="Application not found")

        # Step 2: Extract resume text
        resume_text = extract_resume_text(app_data['resume_path'])
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract resume text")

        # Step 3: Generate first question via Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Based on this resume, generate 1 specific interview question:

Resume (excerpt):
{resume_text[:2000]}

Return ONLY the question text."""
        response = model.generate_content(prompt)
        question_text = response.text.strip()

        # Step 4: Convert question to emotional speech (default neutral for first question)
        audio_base64 = text_to_speech_bytes(question_text) # emotion="neutral"

        # Step 5: Create interview session
        session_id = f"LIVE-{len(interview_sessions) + 1:05d}"
        interview_sessions[session_id] = {
            "application_id": request.application_id,
            "position": app_data['position'],
            "resume_text": resume_text,
            "conversation": [
                {"role": "interviewer", "question": question_text, "emotion": "neutral"}
            ],
            "question_count": 1,
            "max_questions": 4,
            "started_at": datetime.now().isoformat()
        }

        return {
            "session_id": session_id,
            "question": question_text,
            "audio": audio_base64,
            "question_number": 1,
            "total_questions": 4
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interview/continue")
async def continue_live_interview(request: ContinueInterviewRequest):
    """Continue interview: analyze user audio, generate empathetic question, and return emotional TTS"""
    try:
        session = interview_sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Step 1: Analyze candidate audio
        audio_analysis = {}
        empathetic_feedback = ""
        if request.audio_blob_base64:
            audio_analysis = analyze_audio_emotions(request.audio_blob_base64)
            empathetic_feedback = generate_empathetic_response(audio_analysis)

        # Step 2: Store candidate answer
        session['conversation'].append({
            "role": "candidate",
            "answer": request.answer_text,
            "audio_analysis": audio_analysis,
            "timestamp": datetime.now().isoformat()
        })

        # Step 3: Check if maximum questions reached
        if session['question_count'] >= session['max_questions']:
            return await evaluate_live_interview(request.session_id)

        # Step 4: Generate next question using Gemini, incorporating candidate emotion
        conversation_history = "\n".join([
            f"Q: {item['question']}" if 'question' in item else
            f"A: {item['answer']} [Tone: {item.get('audio_analysis', {}).get('tone', 'neutral')}]"
            for item in session['conversation']
        ])
        model = genai.GenerativeModel("gemini-2.5-flash")
        print("Generating prompt through gemini start" )
        prompt = f"""You are an empathetic interviewer.

Resume: {session['resume_text'][:1000]}

Conversation:
{conversation_history}

Candidate tone: {audio_analysis.get('tone', 'neutral')}

Generate next question. Be encouraging if nervous, probe deeper if confident.
Return ONLY the question."""
        response = model.generate_content(prompt)
        question_text = response.text.strip()
        print("Generating prompt through gemini end")

        # Step 5: Combine empathetic feedback + question
        full_response = f"{empathetic_feedback} {question_text}" if empathetic_feedback else question_text

        # Step 6: Convert to emotional speech using ElevenLabs
        detected_tone = audio_analysis.get('tone', 'neutral')
        audio_base64 = text_to_speech_bytes(full_response, audio_data=audio_analysis)

        # Step 7: Update session
        session['conversation'].append({
            "role": "interviewer",
            "question": question_text,
            "empathetic_feedback": empathetic_feedback,
            "emotion": detected_tone
        })
        session['question_count'] += 1

        return {
            "question": question_text,
            "empathetic_feedback": empathetic_feedback,
            "audio": audio_base64,
            "question_number": session['question_count'],
            "total_questions": session['max_questions'],
            "tone_detected": detected_tone,
            "is_final": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def evaluate_live_interview(session_id: str):
    """Evaluate completed interview using Gemini and aggregate audio analytics"""
    try:
        session = interview_sessions[session_id]

        # Step 1: Aggregate tones and confidence
        all_tones = [
            item.get('audio_analysis', {}).get('tone', 'neutral')
            for item in session['conversation'] if item.get('role') == 'candidate'
        ]
        confidence_scores = [
            item.get('audio_analysis', {}).get('confidence_score', 50)
            for item in session['conversation'] if item.get('role') == 'candidate'
        ]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 50

        # Step 2: Build conversation text for evaluation
        conversation_text = "\n\n".join([
            f"Q{i+1}: {session['conversation'][i*2].get('question', '')}\n"
            f"A: {session['conversation'][i*2+1].get('answer', '')}\n"
            f"[Tone: {session['conversation'][i*2+1].get('audio_analysis', {}).get('tone', 'N/A')}]"
            for i in range(len(all_tones))
        ])

        # Step 3: Evaluate with Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""Evaluate this interview:

{conversation_text}

Confidence: {avg_confidence:.1f}%
Tone patterns: {', '.join(set(all_tones))}

Return JSON only:
{{
    "overall_score": 0-100,
    "technical_score": 0-100,
    "confidence_score": 0-100,
    "communication_score": 0-100,
    "strengths": ["list"],
    "areas_for_improvement": ["list"],
    "recommendation": "hire/maybe/reject",
    "summary": "brief assessment"
}}"""
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Cleanup potential code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        evaluation = json.loads(response_text)

        # Step 4: Add audio analytics
        evaluation['audio_analytics'] = {
            "tone_patterns": list(set(all_tones)),
            "dominant_tone": max(set(all_tones), key=all_tones.count) if all_tones else "neutral",
            "avg_confidence": float(avg_confidence)
        }

        # Step 5: Store evaluation
        session['evaluation'] = evaluation
        session['evaluated_at'] = datetime.now().isoformat()

        return {
            "is_final": True,
            "session_id": session_id,
            "evaluation": evaluation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/applications")
async def get_applications():
    return {"total": len(job_applications), "applications": job_applications}


@app.get("/api/interview/sessions")
async def get_all_sessions():
    return {"total": len(interview_sessions), "sessions": interview_sessions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


