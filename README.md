# SMART APPLY 
FastAPI backend for job applications with text-to-speech functionality.

## Features

1. **Job Application Submission** - Accept job applications with resume uploads
2. **Text-to-Speech** - Convert text to audio using ElevenLabs API

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. Submit Job Application

**POST** `/api/job-application`

Submit a job application with resume attachment.

**Form Data:**
- `full_name` (required): Applicant's full name
- `email` (required): Valid email address
- `phone` (required): Phone number
- `position` (required): Position applying for
- `cover_letter` (optional): Cover letter text
- `resume` (required): Resume file (PDF, DOC, DOCX)

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/api/job-application" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "full_name=John Doe" \
  -F "email=john.doe@example.com" \
  -F "phone=+1234567890" \
  -F "position=Software Engineer" \
  -F "cover_letter=I am very interested in this position..." \
  -F "resume=@/path/to/resume.pdf"
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:8000/api/job-application"

files = {
    'resume': open('resume.pdf', 'rb')
}

data = {
    'full_name': 'John Doe',
    'email': 'john.doe@example.com',
    'phone': '+1234567890',
    'position': 'Software Engineer',
    'cover_letter': 'I am very interested in this position...'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Example using JavaScript/Fetch:**

```javascript
const formData = new FormData();
formData.append('full_name', 'John Doe');
formData.append('email', 'john.doe@example.com');
formData.append('phone', '+1234567890');
formData.append('position', 'Software Engineer');
formData.append('cover_letter', 'I am very interested...');
formData.append('resume', fileInput.files[0]); // from <input type="file">

fetch('http://localhost:8000/api/job-application', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response:**

```json
{
  "message": "Application submitted successfully",
  "application_id": "APP-00001",
  "submitted_at": "2025-11-08T10:30:00.123456"
}
```

### 2. Text-to-Speech

**POST** `/api/text-to-speech`

Convert text to speech audio file.

**Request Body (JSON):**

```json
{
  "text": "Hello, this is the interview voice you requested.",
  "voice_id": "pNInz6obpgDQGcFmaJgB",
  "stability": 0.5,
  "similarity_boost": 0.7
}
```

**Parameters:**
- `text` (required): Text to convert to speech
- `voice_id` (optional): ElevenLabs voice ID (default: "pNInz6obpgDQGcFmaJgB")
- `stability` (optional): Voice stability 0-1 (default: 0.5)
- `similarity_boost` (optional): Voice similarity 0-1 (default: 0.7)

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/api/text-to-speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is your interview confirmation.",
    "voice_id": "pNInz6obpgDQGcFmaJgB"
  }' \
  --output speech.mp3
```

**Example using Python requests:**

```python
import requests

url = "http://localhost:8000/api/text-to-speech"

data = {
    "text": "Hello, this is your interview confirmation.",
    "voice_id": "pNInz6obpgDQGcFmaJgB",
    "stability": 0.5,
    "similarity_boost": 0.7
}

response = requests.post(url, json=data)

if response.status_code == 200:
    with open("speech.mp3", "wb") as f:
        f.write(response.content)
    print("Audio saved to speech.mp3")
```

**Example using JavaScript/Fetch:**

```javascript
fetch('http://localhost:8000/api/text-to-speech', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        text: 'Hello, this is your interview confirmation.',
        voice_id: 'pNInz6obpgDQGcFmaJgB'
    })
})
.then(response => response.blob())
.then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'speech.mp3';
    a.click();
});
```

**Response:**
Returns MP3 audio file stream

### 3. Get All Applications (Admin)

**GET** `/api/applications`

Retrieve all submitted applications.

**Example:**

```bash
curl http://localhost:8000/api/applications
```

**Response:**

```json
{
  "total": 2,
  "applications": [
    {
      "application_id": "APP-00001",
      "full_name": "John Doe",
      "email": "john.doe@example.com",
      "phone": "+1234567890",
      "position": "Software Engineer",
      "cover_letter": "I am very interested...",
      "resume_filename": "resume.pdf",
      "resume_size": 102400,
      "submitted_at": "2025-11-08T10:30:00.123456"
    }
  ]
}
```

### 4. Get Specific Application

**GET** `/api/applications/{application_id}`

Retrieve a specific application by ID.

**Example:**

```bash
curl http://localhost:8000/api/applications/APP-00001
```

## Available Voice IDs

Common ElevenLabs voice IDs:
- `pNInz6obpgDQGcFmaJgB` - Adam
- `21m00Tcm4TlvDq8ikWAM` - Rachel
- `AZnzlk1XvdvUeBnXmlld` - Domi
- `EXAVITQu4vr4xnSDxMaL` - Bella
- `ErXwobaYiN019PkySvjV` - Antoni

## Production Considerations

### Security
1. **Add Authentication**: Implement JWT or API key authentication for admin endpoints
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **Input Validation**: Enhanced validation for all inputs
4. **HTTPS**: Use HTTPS in production

### Storage
- Replace in-memory storage with a database (PostgreSQL, MongoDB)
- Store resumes in cloud storage (AWS S3, Google Cloud Storage)
- Add file size limits and virus scanning

### Environment Variables
Create a `.env` file:

```bash
ELEVENLABS_API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:password@localhost/dbname
AWS_ACCESS_KEY=your_aws_key
AWS_SECRET_KEY=your_aws_secret
```

Update code to use environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
```

### Error Handling
- Add comprehensive logging
- Implement proper error responses
- Add retry logic for external API calls

## Testing

Run the API and test with the interactive documentation at `/docs`

## License

MIT
