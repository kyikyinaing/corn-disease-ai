<!-- Project-specific Copilot instructions for AI coding assistants -->
# Copilot instructions — MyCornProject

Purpose
- Help contributors (and AI agents) be productive quickly in this repo: backend FastAPI image diagnostics, a small ML model, a RAG-based assistant, and a Next.js frontend.

Quick high-level architecture
- Backend: `app.py` (FastAPI) exposes `/diagnose` and loads `.env` from the repo root. It writes uploaded images to `uploads/`.
- Prediction: `pipeline.py` composes `ml_model.CornDiseaseModel`, `rag.build_vectorstore('kb')` and `assistant.build_assistant(...)` to produce both model predictions and an assistant answer.
- Core model: `ml_model.py` (expects `corn_model.pth` in project root). A separate helper `torch_model.py` shows an alternate loading pattern and different class-name ordering — prefer `ml_model.py` for production usage.
- Knowledge base + assistant: `rag.py` builds a FAISS vectorstore from `kb/*.txt` using Google Generative embeddings; `assistant.py` wraps a `ChatGoogleGenerativeAI` LLM and retrieves context from the vectorstore.
- Frontend: `corn-ui/` is a Next.js app (dev: `npm run dev`) — main UI is `corn-ui/app/page.tsx` which POSTs to the backend at `http://127.0.0.1:8000/diagnose`.

Critical workflows & run commands
- Run backend (local dev):
  - Start FastAPI with uvicorn from repo root:
    `uvicorn app:app --reload --host 127.0.0.1 --port 8000`
  - Notes: `app.py` loads `.env` (same folder) and sets `KMP_DUPLICATE_LIB_OK=TRUE` for macOS CPU builds.
- Run frontend (Next.js):
  - From `corn-ui/`: `npm run dev` (default port 3000). The UI expects the backend at `http://127.0.0.1:8000`.
- Quick prediction smoke test:
  - `python test_predict.py` (simple runner that imports `CornDiseaseModel` and prints a prediction for a sample image).

Project-specific conventions & gotchas
- Model artifacts:
  - `ml_model.py` requires `corn_model.pth` in the project root; startup will raise `FileNotFoundError` if missing.
  - `torch_model.py` references `corn_resnet18_best.pth` — there are two similar loading styles; prefer `ml_model.py` for production inference.
- Label names:
  - `ml_model.py` defines `CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]` and is the canonical label source for the `pipeline.diagnose` output.
  - `torch_model.py` uses a different naming/ordering. When editing model code, ensure label order stays consistent with the saved weights.
- KB / RAG behavior:
  - `rag.build_vectorstore('kb')` expects plaintext `.txt` files under `kb/` (see `kb/*.txt`). If none are found, it raises `FileNotFoundError`.
  - `rag.py` uses `GoogleGenerativeAIEmbeddings(models/gemini-embedding-001)` and `assistant.py` uses `gemini-2.5-flash`. Ensure relevant Google GenAI credentials/configs are available in the environment when running.
- Assistant API usage:
  - `assistant.build_assistant(vectorstore)` returns a callable `answer(label, confidence, location, notes)`.
  - Note: `assistant.answer` uses `retriever.invoke(query)` and `llm.invoke(prompt)` (these are the library APIs in use here); preserve these call patterns when modifying assistant behavior.
- CORS / host expectations:
  - `app.py` adds CORS allow_origins for `http://localhost:3000`. The frontend fetch uses `http://127.0.0.1:8000` — in development this is expected, but if deploying change allowed origins accordingly.

Files to reference when making edits
- Backend & orchestration: `app.py`, `pipeline.py`
- Model inference: `ml_model.py`, `torch_model.py`, `test_predict.py`
- RAG & assistant: `rag.py`, `assistant.py`, `kb/*.txt`
- Frontend: `corn-ui/app/page.tsx` (example of POST to `/diagnose` and UI expectations)

Editing guidance for AI agents
- When changing prediction labels or their order, update `ml_model.py` and verify `corn_model.pth` was trained with the same label order. Run `python test_predict.py` to sanity-check outputs.
- Preserve the `pipeline.diagnose` signature and returned dict shape: `{'prediction': {'label':..., 'confidence':..., 'all_probabilities':...}, 'assistant_answer': ...}` — the frontend (`corn-ui/app/page.tsx`) depends on this shape.
- Never replace `rag.build_vectorstore` usage with a different embeddings/LLM call without ensuring credentials and vectorstore persistence are handled.
- For quick local dev, prefer CPU runs (the code handles GPU if available). Be mindful of torch device mapping when loading `.pth` files.

If something is missing or unclear
- Tell me which area to expand (deployment, test harness, CI commands, environment variables). Provide missing env values (e.g., which Google GenAI auth method you use) and I will update this document.

---
Updated by AI assistant: drafted from repository files `app.py`, `pipeline.py`, `ml_model.py`, `torch_model.py`, `rag.py`, `assistant.py`, and `corn-ui/app/page.tsx`.
