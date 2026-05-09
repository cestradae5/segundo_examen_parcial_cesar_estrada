# Tutor IA RAG

Sistema de preguntas y respuestas sobre material del curso usando **Retrieval-Augmented Generation (RAG)**. Permite indexar documentos de texto (`.txt`, `.md`, etc.) y consultarlos mediante lenguaje natural, obteniendo respuestas fundamentadas en el contenido indexado.

---

## 🚀 Características

- **Embeddings + LLM**: Utiliza OpenAI (`text-embedding-3-small` para embeddings y `gpt-4o-mini` para generación).
- **Vector Store**: ChromaDB persistido localmente en la carpeta `vectorstore/`.
- **Arquitectura limpia**: Separación de responsabilidades siguiendo Clean Architecture (`domain`, `application`, `infrastructure`, `interfaces`).
- **API REST**: Construida con FastAPI, documentación interactiva disponible en `/docs`.
- **Interfaz de línea de comandos**: Para indexar documentos y lanzar el servidor.
- **Tests unitarios y de integración** con `pytest`.
- **Entorno virtual** recomendado y archivo `requirements.txt`.

---

## 📂 Estructura del proyecto

```
tutor-ia-rag/
├── docs/curso_ia/          # ← Coloca aquí los documentos del curso (.txt, .md)
├── vectorstore/            # ← Generado automáticamente (no subir a git)
├── src/
│   ├── domain/             # Entidades puras y puertos
│   ├── application/        # Casos de uso (indexar, consultar)
│   ├── infrastructure/     # Implementaciones: ChromaDB, OpenAI, cargadores, índice
│   └── interfaces/         # FastAPI (endpoints) y CLI
├── tests/
│   ├── unit/               # Tests unitarios
│   └── integration/        # Tests de integración
├── main.py                 # Punto de entrada (CLI)
├── requirements.txt        # Dependencias de Python
├── .env.example            # Ejemplo de variables de entorno
└── README.md
```

---

## ⚙️ Setup (Instalación)

1. **Clona el repositorio** (si no lo has hecho ya):

   ```bash
   git clone https://github.com/cestradae5/segundo_examen_parcial_cesar_estrada.git
   cd segundo_examen_parcial_cesar_estrada/tutor-ia-rag
   ```

2. **Crea y activa un entorno virtual**:

   ```bash
   # Crear entorno virtual
   python -m venv .venv

   # Activarlo
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Instala las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configura las variables de entorno**:

   Copia el archivo de ejemplo y edítalo con tu clave de API de OpenAI:

   ```bash
   copy .env.example .env   # Windows
   # o
   cp .env.example .env     # macOS / Linux
   ```

   Edita `.env` y agrega:

   ```
   OPENAI_API_KEY=sk-tuaqui...
   ```

   (Opcional) Puedes ajustar otros parámetros como el modelo de embedding o la ruta del vectorstore.

---

## 🛠️ Uso

### 1. Indexar los documentos del curso

Coloca los archivos de texto (`.txt`, `.md`, etc.) dentro de la carpeta `docs/curso_ia/`. Luego ejecuta:

```bash
python main.py index --docs docs/curso_ia
```

Este proceso:
- Lee cada documento.
- Lo divide en fragmentos (chunks).
- Calcula embeddings con OpenAI.
- Guarda los vectores en ChromaDB (en `vectorstore/`).

> **Nota**: La primera ejecución puede tardar unos segundos según la cantidad y tamaño de los documentos.

### 2. Levantar la API

```bash
python main.py serve
```

La API estará disponible en `http://localhost:8000`.

- Documentación interactiva (Swagger UI): http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Realizar consultas

Puedes usar la interfaz de Swagger (`/docs`) o hacer una petición `POST` al endpoint `/ask`.

#### Ejemplo con `curl`:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es el aprendizaje supervisado?", "top_k": 5}'
```

#### Respuesta esperada (JSON):

```json
{
  "answer": "El aprendizaje supervisado es una rama del aprendizaje automático donde el modelo se entrena utilizando datos etiquetados...",
  "sources": [
    {
      "id": "doc1.txt",
      "score": 0.87,
      "excerpt": "En el aprendizaje supervisado, cada ejemplo de entrenamiento consiste en un vector de características y una etiqueta..."
    },
    // ... más fuentes
  ]
}
```

- `answer`: Respuesta generada por el LLM basada en los fragmentos recuperados.
- `sources`: Lista de los documentos/fragmentos más relevantes utilizados, con su puntuación de similitud y un fragmento de texto.

### 4. Parámetros del endpoint `/ask`

| Parámetro | Tipo   | Descripción                                   | Default |
|-----------|--------|-----------------------------------------------|---------|
| `question`| string | Pregunta en lenguaje natural.                 | –       |
| `top_k`   | int    | Número de fragmentos a recuperar antes de generar la respuesta. | 4       |

---

## 🧪 Ejecutar tests

Para asegurarte de que todo funciona correctamente:

```bash
pytest tests/
```

Los tests cubren:
- Entidades del dominio.
- Casos de uso (indexado y consulta).
- Adaptadores de infraestructura (ChromaDB, cargadores de texto).
- Endpoints de la API.

---

## 📦 Dependencias principales

- `fastapi`
- `uvicorn`
- `openai`
- `chromadb`
- `python-dotenv`
- `pytest`
- `pytest-asyncio`

Ver el archivo completo `requirements.txt` para versiones exactas.

---

## 🤝 Contribuir

1. Haz un **fork** del repositorio.
2. Crea una rama para tu feature o bugfix:  
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Realiza tus cambios y escribe tests si corresponde.
4. Asegúrate de que los tests pasen: `pytest`.
5. Haz commit con un mensaje descriptivo.
7. Push a tu fork y abre un **Pull Request** hacia `main`.

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- [OpenAI](https://openai.com) por los modelos de embeddings y lenguaje.
- [ChromaDB](https://www.trychroma.com) por el vector store sencillo y potente.
- [FastAPI](https://fastapi.tiangolo.com) por el framework web rápido y moderno.
- La comunidad de Python por las herramientas que hacen posible este tipo de aplicaciones.

---

**¡Disfruta aprendiendo con tu tutor IA!** Si tienes preguntas o sugerencias, abre un issue o contáctanos.
