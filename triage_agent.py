from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

def triage_questions(label: str, confidence: float) -> List[Dict[str, str]]:
    qs = []

    # Always ask location (helps advice)
    qs.append({"id": "location", "q": "Where is your farm (country/city)?", "type": "text"})

    # Confidence-based photo request
    if confidence < 0.60:
        qs.append({"id": "photo_tip", "q": "Please upload 1 more photo: (1) close-up spot, (2) full leaf in daylight. Can you do that?", "type": "yesno"})

    # Disease-specific questions
    if label != "Healthy":
        qs.extend([
            {"id": "severity", "q": "About how many plants are affected? (few / some / many)", "type": "choice"},
            {"id": "spread", "q": "Is it spreading fast in the last 3–7 days? (yes/no)", "type": "yesno"},
            {"id": "weather", "q": "Was there heavy rain, dew, or high humidity recently? (yes/no)", "type": "yesno"},
        ])
    else:
        qs.append({"id": "symptoms", "q": "Do you still see spots, rust powder, or large brown dead areas? (yes/no)", "type": "yesno"})

    return qs[:5]  # keep it short

def build_triage_agent(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def run(label: str, confidence: float, language: str, user_answers: Dict[str, Any]) -> str:
        # Retrieve KB context
        docs = retriever.invoke(f"{label} corn disease symptoms treatment prevention")
        context = "\n\n".join(d.page_content for d in docs) if docs else "No KB context available."

        # Build a refined prompt using answers
        prompt = f"""
Reply in: {language}

You are a helpful agriculture assistant for corn leaf disease.
Use KB context first. If KB is missing, give general best-practice advice and clearly say it is general.

Prediction:
- Disease: {label}
- Confidence: {confidence:.3f}

User answers:
{user_answers}

KB Context:
{context}

Now produce a refined response:
1) Short summary (1-2 lines)
2) How to confirm (3 bullet points)
3) What to do now (step-by-step)
4) Prevention tips
5) If confidence < 0.60 include a warning: take clearer photos / consult an expert
Keep language simple and user-friendly.
"""
        resp = llm.invoke(prompt)
        return resp.content

    return run