
import requests
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Local AI Study Buddy")

client = chromadb.PersistentClient(path="./study_data")
notes = client.get_or_create_collection(name="study_notes")

MODEL = "gemma3:1b"


class Note(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
    <title>AI Study Buddy</title>
    <style>
    body { font-family: Arial; max-width: 750px; margin: 40px auto;
           padding: 20px; background: #f1f3ff; }
    section { background: white; padding: 20px; margin: 20px 0;
              border-radius: 12px; }
    input, textarea, button { width: 100%; padding: 12px;
                              margin: 8px 0; box-sizing: border-box; }
    button { background: #5145cd; color: white; border: 0;
             cursor: pointer; }
    pre { white-space: pre-wrap; }
    </style>
    </head>
    <body>
    <h1>📚 Local AI Study Buddy</h1>
    <p>Your personal AI powered by a local model.</p>
    <section>
    <h2>Add Study Notes</h2>
    <input id="title" placeholder="Note title">
    <textarea id="content" placeholder="Paste your notes here"></textarea>
    <button onclick="saveNote()">Save Notes</button>
    <p id="saved"></p>
    </section>
    <section>
    <h2>Ask Your AI</h2>
    <textarea id="question" placeholder="Ask a question"></textarea>
    <button onclick="askAI()">Get Answer</button>
    <p id="status"></p>
    <pre id="answer"></pre>
    </section>
    <script>
    async function saveNote() {
      const title = document.getElementById('title').value;
      const content = document.getElementById('content').value;
      const r = await fetch('/add-note', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({title, content})
      });
      const d = await r.json();
      document.getElementById('saved').textContent =
        r.ok ? d.message : d.detail;
    }
    async function askAI() {
      document.getElementById('status').textContent = 'Thinking...';
      const question = document.getElementById('question').value;
      try {
        const r = await fetch('/ask', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({question})
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail);
        document.getElementById('answer').textContent =
          d.answer + '\\n\\nSources: ' + d.sources.join(', ');
        document.getElementById('status').textContent = 'Answer ready!';
      } catch(e) {
        document.getElementById('status').textContent = e.message;
      }
    }
    </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "running", "saved_notes": notes.count()}


@app.post("/add-note")
def add_note(note: Note):
    title = note.title.strip()
    content = note.content.strip()

    if not title or not content:
        raise HTTPException(400, "Enter a title and some notes.")

    import uuid
    notes.add(
        ids=[uuid.uuid4().hex],
        documents=[content],
        metadatas=[{"title": title}]
    )
    return {"message": "Notes saved successfully!"}


@app.post("/ask")
def ask(question: Question):
    text = question.question.strip()

    if not text:
        raise HTTPException(400, "Enter a question.")

    if notes.count() == 0:
        raise HTTPException(400, "Save some study notes first.")

    results = notes.query(
        query_texts=[text],
        n_results=min(3, notes.count())
    )

    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    context = "\n\n".join(
        f"{metadata[i]['title']}: {docs[i]}"
        for i in range(len(docs))
    )

    prompt = f"""
Answer the student's question using the notes below.
Explain in simple language. If the notes don't contain
the answer, say so clearly.

NOTES:
{context}

QUESTION:
{text}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=300
        )
        response.raise_for_status()
        answer = response.json()["response"]
    except requests.RequestException:
        raise HTTPException(
            503, "Ollama is unavailable. Start Ollama and try again."
        )

    sources = list(dict.fromkeys(m["title"] for m in metadata))
    return {"answer": answer, "sources": sources}
