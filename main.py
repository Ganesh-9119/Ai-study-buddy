import uuid
import requests
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="AI Study Buddy")

# Store notes locally
db = chromadb.PersistentClient(path="./study_data")
notes = db.get_or_create_collection(name="study_notes")

MODEL = "gemma3:1b"


class Note(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str


PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Study Buddy</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 750px;
            margin: 30px auto;
            padding: 20px;
            background: #f0f0ff;
            color: #222;
        }
        h1 { color: #5145cd; }
        section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 12px;
        }
        input, textarea, button {
            box-sizing: border-box;
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            font-size: 16px;
        }
        textarea { min-height: 100px; }
        button {
            background: #5145cd;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        pre { white-space: pre-wrap; overflow-wrap: anywhere; }
    </style>
</head>
<body>
    <h1>📚 AI Study Buddy</h1>
    <p>Your personal study assistant, powered by local AI.</p>

    <section>
        <h2>Save Study Notes</h2>
        <input id="title" placeholder="Note title">
        <textarea id="content"
            placeholder="Paste your study notes here"></textarea>
        <button onclick="saveNote()">Save Notes</button>
        <p id="saveStatus"></p>
    </section>

    <section>
        <h2>Ask Your Notes</h2>
        <textarea id="question"
            placeholder="Example: Explain a Python loop"></textarea>
        <button onclick="askAI()">Ask AI</button>
        <p id="status"></p>
        <pre id="answer"></pre>
    </section>

    <script>
    async function saveNote() {
        const title = document.getElementById("title").value.trim();
        const content = document.getElementById("content").value.trim();
        const status = document.getElementById("saveStatus");

        if (!title || !content) {
            status.textContent = "Enter a title and notes.";
            return;
        }

        status.textContent = "Saving...";
        try {
            const r = await fetch("/add-note", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({title, content})
            });
            const d = await r.json();
            status.textContent = r.ok ? d.message : d.detail;
        } catch (e) {
            status.textContent = "Error: " + e.message;
        }
    }

    async function askAI() {
        const question = document.getElementById("question").value.trim();
        const status = document.getElementById("status");
        const answer = document.getElementById("answer");

        if (!question) {
            status.textContent = "Please enter a question.";
            return;
        }

        status.textContent = "AI is thinking...";
        answer.textContent = "";

        try {
            const r = await fetch("/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question})
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail);

            answer.textContent = d.answer +
                "\\n\\nSources: " + d.sources.join(", ");
            status.textContent = "Answer ready!";
        } catch (e) {
            status.textContent = "Error: " + e.message;
        }
    }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"status": "running", "saved_notes": notes.count()}


@app.post("/add-note")
def add_note(note: Note):
    title = note.title.strip()
    content = note.content.strip()

    if not title or not content:
        raise HTTPException(400, "Enter a title and note content.")

    notes.add(
        ids=[uuid.uuid4().hex],
        documents=[content],
        metadatas=[{"title": title}]
    )

    return {"message": "Study notes saved successfully!"}


@app.post("/ask")
def ask(item: Question):
    question = item.question.strip()

    if not question:
        raise HTTPException(400, "Enter a question.")

    if notes.count() == 0:
        raise HTTPException(400, "Save your study notes first.")

    try:
        results = notes.query(
            query_texts=[question],
            n_results=min(3, notes.count())
        )

        documents = results["documents"][0]
        metadata = results["metadatas"][0]

        context = "\n\n".join(
            f"Title: {metadata[i]['title']}\n{documents[i]}"
            for i in range(len(documents))
        )

        prompt = f"""
You are a friendly college study assistant.
Answer simply using the student's saved notes.
If the notes do not contain the answer, say so clearly.

STUDY NOTES:
{context}

STUDENT QUESTION:
{question}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()

        answer = response.json()["response"]
        sources = list(dict.fromkeys(m["title"] for m in metadata))

        return {"answer": answer, "sources": sources}

    except requests.RequestException:
        raise HTTPException(
            503,
            "Ollama is unavailable. Make sure Ollama is running."
        )
    except Exception as e:
        raise HTTPException(500, f"Could not answer question: {e}")