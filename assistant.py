from langchain_google_genai import ChatGoogleGenerativeAI

def build_assistant(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def answer(label: str, confidence: float, location: str = "", notes: str = "", language: str = "English") -> str:
        query = f"{label} corn disease symptoms treatment prevention"
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs) if docs else "No context found in knowledge base."

        prompt = f"""
You are an agriculture assistant for corn leaf disease.
Reply in: {language}

Use the context first. If the context is missing, give general best-practice advice and clearly say it is general.

Prediction:
- Disease: {label}
- Confidence: {confidence:.3f}

User info:
- Location: {location}
- Notes: {notes}

Context:
{context}

Task:
1) Explain the disease in very simple words.
2) List 3 symptoms to confirm.
3) Give treatment steps (safe + practical).
4) Give prevention tips.
5) If confidence < 0.60 add a warning: take clearer photos / consult an expert.
"""
        resp = llm.invoke(prompt)
        return resp.content

    return answer