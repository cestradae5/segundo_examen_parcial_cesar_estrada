import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.application.use_cases import AskTutorUC, IndexDocumentsUC
from src.domain.entities import Query
from src.infrastructure.indexer import DocumentIndexer
from src.infrastructure.rag_chain import RAGChain
from src.infrastructure.vector_store import ChromaVectorStore

_DOCS_PATH = Path("docs/curso_ia")
_ALLOWED_EXTENSIONS = {".pdf", ".txt"}

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

    /* ── header ── */
    header {
      width: 100%;
      max-width: 760px;
      padding: 1.25rem 1rem 0;
    }
    header h1 { font-size: 1.3rem; font-weight: 700; color: #7dd3fc; }
    header p  { font-size: 0.8rem; color: #64748b; margin-top: 2px; }

    /* ── panel de documentos ── */
    #docs-panel {
      width: 100%;
      max-width: 760px;
      margin: .75rem 1rem 0;
      padding: .75rem 1rem;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 10px;
    }
    #docs-toggle {
      width: 100%;
      background: none;
      border: none;
      color: #94a3b8;
      font-size: .82rem;
      text-align: left;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: .4rem;
    }
    #docs-toggle:hover { color: #e2e8f0; }
    #docs-toggle .arrow { transition: transform .2s; display: inline-block; }
    #docs-toggle.open .arrow { transform: rotate(90deg); }

    #docs-body { display: none; margin-top: .75rem; }
    #docs-body.open { display: block; }

    #drop-zone {
      border: 2px dashed #334155;
      border-radius: 8px;
      padding: 1.2rem;
      text-align: center;
      color: #64748b;
      font-size: .85rem;
      cursor: pointer;
      transition: border-color .15s, background .15s;
    }
    #drop-zone.dragover, #drop-zone:hover { border-color: #3b82f6; background: #1d3352; color: #93c5fd; }
    #file-input { display: none; }

    #file-list {
      margin-top: .5rem;
      font-size: .8rem;
      color: #94a3b8;
      min-height: 1.2rem;
    }

    .docs-actions {
      display: flex;
      gap: .5rem;
      margin-top: .65rem;
    }
    .docs-actions button {
      padding: .45rem .85rem;
      border-radius: 7px;
      border: none;
      font-size: .82rem;
      font-weight: 600;
      cursor: pointer;
      transition: background .15s;
    }
    #btn-upload { background: #0369a1; color: #fff; }
    #btn-upload:hover:not(:disabled) { background: #0284c7; }
    #btn-index  { background: #065f46; color: #fff; }
    #btn-index:hover:not(:disabled)  { background: #047857; }
    .docs-actions button:disabled { background: #334155; color: #64748b; cursor: default; }

    #docs-status {
      margin-top: .5rem;
      font-size: .78rem;
      min-height: 1rem;
      color: #94a3b8;
    }
    #docs-status.ok    { color: #34d399; }
    #docs-status.error { color: #f87171; }

    /* ── chat ── */
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
    .msg.error   { background: #7f1d1d; border-color: #dc2626; }
    .msg.thinking { color: #64748b; font-style: italic; }

    /* ── input form ── */
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
    #btn-ask {
      padding: .65rem 1.1rem;
      border-radius: 8px;
      border: none;
      background: #3b82f6;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
      transition: background .15s;
    }
    #btn-ask:hover:not(:disabled) { background: #2563eb; }
    #btn-ask:disabled { background: #334155; cursor: default; }
  </style>
</head>
<body>
  <header>
    <h1>Tutor IA RAG</h1>
    <p>Hacé preguntas sobre el material del curso.</p>
  </header>

  <!-- Panel de documentos -->
  <div id="docs-panel">
    <button id="docs-toggle">
      <span class="arrow">▶</span> Gestionar documentos del curso
    </button>
    <div id="docs-body">
      <div id="drop-zone">
        Arrastrá archivos acá o hacé click para seleccionar<br/>
        <small>PDF y TXT — podés subir varios a la vez</small>
      </div>
      <input id="file-input" type="file" multiple accept=".pdf,.txt" />
      <div id="file-list"></div>
      <div class="docs-actions">
        <button id="btn-upload" disabled>Subir archivos</button>
        <button id="btn-index">Indexar documentos</button>
      </div>
      <div id="docs-status"></div>
    </div>
  </div>

  <div id="chat"></div>

  <form id="form">
    <textarea id="input" rows="1" placeholder="¿Qué es backpropagation?" required></textarea>
    <button id="btn-ask" type="submit">Enviar</button>
  </form>

  <script>
    // ── panel toggle ──
    const toggle   = document.getElementById('docs-toggle');
    const docsBody = document.getElementById('docs-body');
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('open');
      docsBody.classList.toggle('open');
    });

    // ── file picker ──
    const dropZone  = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList  = document.getElementById('file-list');
    const btnUpload = document.getElementById('btn-upload');
    const btnIndex  = document.getElementById('btn-index');
    const docsStatus = document.getElementById('docs-status');

    let selectedFiles = [];

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      setFiles([...e.dataTransfer.files]);
    });
    fileInput.addEventListener('change', () => setFiles([...fileInput.files]));

    function setFiles(files) {
      selectedFiles = files.filter(f => f.name.endsWith('.pdf') || f.name.endsWith('.txt'));
      if (selectedFiles.length === 0) {
        fileList.textContent = 'Solo se aceptan archivos .pdf y .txt';
        btnUpload.disabled = true;
        return;
      }
      fileList.textContent = selectedFiles.map(f => f.name).join(', ');
      btnUpload.disabled = false;
    }

    function setStatus(msg, type) {
      docsStatus.textContent = msg;
      docsStatus.className = type || '';
    }

    // ── upload ──
    btnUpload.addEventListener('click', async () => {
      if (!selectedFiles.length) return;
      btnUpload.disabled = true;
      setStatus('Subiendo archivos...', '');

      const fd = new FormData();
      selectedFiles.forEach(f => fd.append('files', f));

      try {
        const res  = await fetch('/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) { setStatus('Error: ' + (data.detail || res.statusText), 'error'); return; }
        setStatus(`Subidos: ${data.uploaded.join(', ')}. Ahora podés indexarlos.`, 'ok');
        fileList.textContent = '';
        selectedFiles = [];
      } catch (err) {
        setStatus('No se pudo conectar con el servidor.', 'error');
      } finally {
        btnUpload.disabled = true;
      }
    });

    // ── index ──
    btnIndex.addEventListener('click', async () => {
      btnIndex.disabled = true;
      setStatus('Indexando documentos… esto puede tardar un momento.', '');
      try {
        const res  = await fetch('/index', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) { setStatus('Error: ' + (data.detail || res.statusText), 'error'); return; }
        setStatus(`Indexados ${data.chunks} chunks correctamente.`, 'ok');
      } catch (err) {
        setStatus('No se pudo conectar con el servidor.', 'error');
      } finally {
        btnIndex.disabled = false;
      }
    });

    // ── chat ──
    const chat   = document.getElementById('chat');
    const form   = document.getElementById('form');
    const input  = document.getElementById('input');
    const btnAsk = document.getElementById('btn-ask');

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
      btnAsk.disabled = true;

      const thinking = addMsg('Pensando...', 'bot thinking');

      try {
        const res  = await fetch('/ask', {
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
        btnAsk.disabled = false;
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


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    _DOCS_PATH.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido: {file.filename}. Solo PDF y TXT.",
            )
        dest = _DOCS_PATH / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded.append(file.filename)

    if not uploaded:
        raise HTTPException(status_code=400, detail="No se recibieron archivos válidos.")

    return {"uploaded": uploaded}


@app.post("/index")
def run_index():
    if not _DOCS_PATH.exists() or not any(_DOCS_PATH.iterdir()):
        raise HTTPException(
            status_code=400,
            detail="No hay documentos en docs/curso_ia/. Subí archivos primero.",
        )
    indexer = DocumentIndexer()
    uc = IndexDocumentsUC(indexer)
    try:
        total = uc.execute()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"chunks": total}


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
