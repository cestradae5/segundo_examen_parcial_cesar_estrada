from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.application.use_cases import AskTutorUC
from src.domain.entities import Query
from src.infrastructure.rag_chain import RAGChain
from src.infrastructure.vector_store import ChromaVectorStore

app = FastAPI(title="Tutor IA RAG", version="1.0.0")

_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tutor IA RAG</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      height: 100dvh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    header {
      width: 100%;
      max-width: 760px;
      padding: 1.25rem 1rem 0;
    }
    header h1 { font-size: 1.3rem; font-weight: 700; color: #7dd3fc; }
    header p  { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

    #chat {
      flex: 1;
      width: 100%;
      max-width: 760px;
      overflow-y: auto;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .msg {
      max-width: 88%;
      padding: .75rem 1rem;
      border-radius: 12px;
      line-height: 1.55;
      font-size: .9rem;
      white-space: pre-wrap;
    }
    .msg.user {
      align-self: flex-end;
      background: #1d4ed8;
      color: #fff;
      border-bottom-right-radius: 3px;
    }
    .msg.bot {
      align-self: flex-start;
      background: #1e293b;
      border: 1px solid #334155;
      border-bottom-left-radius: 3px;
    }
    .msg.bot .sources {
      margin-top: .5rem;
      font-size: .75rem;
      color: #94a3b8;
      border-top: 1px solid #334155;
      padding-top: .4rem;
    }
    .msg.error { background: #7f1d1d; border-color: #dc2626; }
    .msg.thinking { color: #64748b; font-style: italic; }

    #form {
      width: 100%;
      max-width: 760px;
      padding: .75rem 1rem 1.25rem;
      display: flex;
      gap: .5rem;
    }
    #input {
      flex: 1;
      padding: .65rem .9rem;
      border-radius: 8px;
      border: 1px solid #334155;
      background: #1e293b;
      color: #e2e8f0;
      font-size: .9rem;
      resize: none;
      outline: none;
    }
    #input:focus { border-color: #3b82f6; }
    #btn {
      padding: .65rem 1.1rem;
      border-radius: 8px;
      border: none;
      background: #3b82f6;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      transition: background .15s;
    }
    #btn:hover:not(:disabled) { background: #2563eb; }
    #btn:disabled { background: #334155; cursor: default; }
  </style>
</head>
<body>
  <header>
    <h1>Tutor IA RAG</h1>
    <p>Hacé preguntas sobre el material del curso. Responde solo con lo indexado.</p>
  </header>

  <div id="chat"></div>

  <form id="form">
    <textarea id="input" rows="1" placeholder="¿Qué es backpropagation?" required></textarea>
    <button id="btn" type="submit">Enviar</button>
  </form>

  <script>
    const chat  = document.getElementById('chat');
    const form  = document.getElementById('form');
    const input = document.getElementById('input');
    const btn   = document.getElementById('btn');

    function addMsg(content, role, sources) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = content;
      if (sources && sources.length) {
        const s = document.createElement('div');
        s.className = 'sources';
        const unique = [...new Set(sources.map(p => p.split(/[\\\\/]/).pop()))];
        s.textContent = 'Fuentes: ' + unique.join(', ');
        div.appendChild(s);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });

    form.addEventListener('submit', async e => {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;

      addMsg(q, 'user');
      input.value = '';
      btn.disabled = true;

      const thinking = addMsg('Pensando...', 'bot thinking');

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, top_k: 5 }),
        });

        const data = await res.json();
        thinking.remove();

        if (!res.ok) {
          addMsg('Error: ' + (data.detail || res.statusText), 'bot error');
        } else {
          addMsg(data.answer, 'bot', data.sources);
        }
      } catch (err) {
        thinking.remove();
        addMsg('No se pudo conectar con el servidor.', 'bot error');
      } finally {
        btn.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>"""


def _build_use_case() -> AskTutorUC:
    store = ChromaVectorStore()
    chain = RAGChain(store)
    return AskTutorUC(chain)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponseDTO(BaseModel):
    answer: str
    sources: list[str]
    query: str


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


@app.post("/ask", response_model=AskResponseDTO)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    uc = _build_use_case()
    query = Query(text=request.question, top_k=request.top_k)
    result = uc.execute(query)

    return AskResponseDTO(
        answer=result.answer,
        sources=[chunk.source for chunk in result.sources],
        query=result.query,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
