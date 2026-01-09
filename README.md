# RAG Chatbot with Supabase and FastAPI

A production-ready Retrieval Augmented Generation (RAG) chatbot API that enables semantic search over PDF documents using OpenAI embeddings and Supabase vector storage.

## Features

- **PDF Document Ingestion**: Upload and automatically process PDF files into searchable chunks
- **Vector Similarity Search**: Semantic search using OpenAI embeddings (1536-dim) and Supabase pgvector
- **AI-Powered Responses**: Context-aware answers using GPT-4o-mini with retrieved document chunks
- **RESTful API**: Clean FastAPI endpoints with automatic OpenAPI documentation
- **Scalable Architecture**: Modular design with separation of concerns (routes → services → clients)

## Architecture
```
RAG Pipeline Flow:
1. PDF Upload → PyPDFLoader
2. Text Chunking → CharacterTextSplitter (1000 chars, 200 overlap)
3. Embedding → OpenAI text-embedding-3-small
4. Storage → Supabase (PostgreSQL + pgvector)
5. Query → Embed query → Similarity search → LLM generation
```

**Tech Stack:**
- **Backend**: FastAPI, Uvicorn
- **Vector Store**: Supabase (PostgreSQL + pgvector extension)
- **AI/ML**: OpenAI API (GPT-4o-mini, text-embedding-3-small)
- **Document Processing**: LangChain, PyPDF

## 📁 Project Structure
```
test-app-name/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── core/
│   │   ├── config.py           # Pydantic settings management
│   │   └── logging.py          # Logging configuration
│   ├── clients/
│   │   ├── openai_client.py    # OpenAI client wrapper
│   │   ├── supabase_client.py  # Supabase client wrapper
│   │   └── embeddings.py       # Embeddings interface
│   ├── services/
│   │   ├── ingest_service.py   # PDF processing & vectorization
│   │   ├── retrieval_service.py # Semantic search via RPC
│   │   └── chat_service.py     # LLM response generation
│   ├── api/routes/
│   │   ├── health.py           # Health check endpoint
│   │   ├── ingest.py           # Document upload endpoint
│   │   └── chat.py             # Chat/query endpoint
│   └── models/
│       └── schemas.py          # Pydantic request/response models
├── tmp/                        # Temporary file storage
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.12+
- OpenAI API key (with billing enabled)
- Supabase account

### 1. Clone Repository
```bash
git clone git@github.com:yourusername/test-app-name.git
cd test-app-name
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Supabase

Create a new Supabase project and run these SQL commands:
```sql
-- Enable pgvector extension
create extension if not exists vector;

-- Create chunks table
create table chunks (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(1536)
);

-- Create similarity search function
create or replace function public.match_chunks (
  query_embedding vector(1536),
  match_threshold float default 0.0,
  match_count int default 5,
  filter jsonb default '{}'::jsonb
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select
    c.id,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  where c.metadata @> filter
    and (1 - (c.embedding <=> query_embedding)) >= match_threshold
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
```

### 5. Environment Configuration

Create `.env` file:
```bash
OPENAI_API_KEY=your-openai-api-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

### 6. Run Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access Swagger UI at: http://127.0.0.1:8000/docs

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Upload PDF
```bash
POST /ingest/file
Content-Type: multipart/form-data

Parameters:
- file: PDF file
- chunk_size: int (default: 1000)
- chunk_overlap: int (default: 200)
- keep_file: bool (default: false)
```

Example:
```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -F 'file=@document.pdf'
```

### Chat Query
```bash
POST /chat
Content-Type: application/json

Body:
{
  "query": "Your question here",
  "k": 5,
  "filter": {},
  "match_threshold": 0.0,
  "model": "gpt-4o-mini",
  "max_output_tokens": 400,
  "temperature": 0.4
}
```

Example:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "k": 4
  }'
```

## 🔧 Configuration

Key settings in `app/core/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `openai_model` | gpt-4o-mini | Chat completion model |
| `max_k` | 20 | Max retrieval results |
| `default_temperature` | 0.4 | LLM temperature |

## Testing
```bash
# Health check
curl http://localhost:8000/health

# Upload test document
curl -X POST "http://localhost:8000/ingest/file" \
  -F 'file=@test.pdf'

# Query
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the main points", "k": 5}'
```

## Known Issues

- OpenAI API rate limiting on free tier (requires billing setup)
- Large PDFs may take time to process (consider async job queue for production)

## Future Enhancements

- [ ] Add user authentication (JWT)
- [ ] Implement conversation history/memory
- [ ] Support additional file formats (DOCX, TXT, MD)
- [ ] Add caching layer (Redis)
- [ ] Query rewriting for better retrieval
- [ ] Deploy to cloud (Docker + AWS/GCP)
- [ ] Add monitoring and analytics

## License

MIT

## Author

Corey Zhang - Northwestern University

## Acknowledgments

- Tutorial based on [Creating a RAG Chatbot with Supabase + OpenAI](https://ayeshasahar.hashnode.dev/creating-a-rag-chatbot-with-supabase-openai-python-langchain) provided by Hunter Zhao from GPT-trainer
