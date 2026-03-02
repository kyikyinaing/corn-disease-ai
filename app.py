import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
import json
from pipeline import triage


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Subprocess-safe: load .env from the same folder as app.py
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from pipeline import diagnose  # after dotenv load

app = FastAPI()
os.makedirs("uploads", exist_ok=True)

@app.post("/diagnose")
async def diagnose_endpoint(
    image: UploadFile = File(...),
    location: str = Form(""),
    notes: str = Form(""),
    language: str = Form("English"),
):
    safe_name = image.filename.replace("/", "_").replace("\\", "_")
    file_path = os.path.join("uploads", f"{uuid.uuid4()}_{safe_name}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    return diagnose(file_path, location=location, notes=notes,language=language)

@app.get("/")
def home():
    return {"status": "ok", "message": "Corn disease API is running. Go to /docs"}

from fastapi import HTTPException

@app.post("/diagnose")
async def diagnose_endpoint(
    image: UploadFile = File(...),
    location: str = Form(""),
    notes: str = Form("")
):
    try:
        safe_name = image.filename.replace("/", "_").replace("\\", "_")
        file_path = os.path.join("uploads", f"{uuid.uuid4()}_{safe_name}")

        with open(file_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        return diagnose(file_path, location=location, notes=notes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # later you can add your deployed domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/triage")
async def triage_endpoint(
    image: UploadFile = File(...),
    language: str = Form("English"),
    answers_json: str = Form(""),  # send answers as JSON string
):
    safe_name = image.filename.replace("/", "_").replace("\\", "_")
    file_path = os.path.join("uploads", f"{uuid.uuid4()}_{safe_name}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    answers = {}
    if answers_json.strip():
        answers = json.loads(answers_json)

    return triage(file_path, language=language, answers=answers)