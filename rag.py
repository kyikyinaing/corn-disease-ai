import os
import glob
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_vectorstore(kb_folder: str = "kb"):
    # Load .env here (SAFE place)
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Gemini API key not found. Set GOOGLE_API_KEY in .env")

    files = glob.glob(os.path.join(kb_folder, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in '{kb_folder}'")

    docs = []
    for f in files:
        docs.extend(TextLoader(f, encoding="utf-8").load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=api_key
    )

    return FAISS.from_documents(chunks, embeddings)