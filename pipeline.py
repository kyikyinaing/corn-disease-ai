from ml_model import CornDiseaseModel
from rag import build_vectorstore
from assistant import build_assistant
from triage_agent import triage_questions, build_triage_agent

model = CornDiseaseModel()
vectorstore = build_vectorstore("kb")

assistant = build_assistant(vectorstore)          # existing final advice
triage_agent = build_triage_agent(vectorstore)    # new refined advice

def diagnose(image_path: str, location: str = "", notes: str = "", language: str = "English") -> dict:
    pred = model.predict_from_image(image_path)
    label = pred["label"]
    confidence = float(pred["confidence"])

    advice = assistant(label, confidence, location=location, notes=notes, language=language)

    return {
        "prediction": {
            "label": label,
            "confidence": round(confidence, 3),
            "all_probabilities": pred.get("all_probabilities", {})
        },
        "assistant_answer": advice
    }

def triage(image_path: str, language: str = "English", answers: dict | None = None) -> dict:
    answers = answers or {}

    pred = model.predict_from_image(image_path)
    label = pred["label"]
    confidence = float(pred["confidence"])

    # If no answers yet, ask questions first
    if not answers:
        return {
            "prediction": {
                "label": label,
                "confidence": round(confidence, 3),
                "all_probabilities": pred.get("all_probabilities", {})
            },
            "mode": "questions",
            "questions": triage_questions(label, confidence),
        }

    # If answers provided, return refined advice
    refined = triage_agent(label, confidence, language=language, user_answers=answers)

    return {
        "prediction": {
            "label": label,
            "confidence": round(confidence, 3),
            "all_probabilities": pred.get("all_probabilities", {})
        },
        "mode": "refined_advice",
        "assistant_answer": refined
    }