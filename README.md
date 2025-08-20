# Vivum RAG Backend

A production-ready Retrieval-Augmented Generation (RAG) system for researchers to query PubMed articles with AI-powered insights. Built with FastAPI, LangChain, and FAISS for efficient medical literature analysis.

## 🚀 Features

- **PubMed Integration**: Fetch and process research articles from PubMed with advanced filtering
- **Topic Isolation**: Each research topic has its own isolated vector store preventing cross-contamination
- **Conversational AI**: Multi-turn conversations with context retention using LangChain
- **Advanced Search**: Support for boolean operators (AND, OR, NOT) and complex queries
- **Performance Optimized**: Batch processing for fast article ingestion (~30 seconds for 20 articles)
- **Resource Management**: Automatic cleanup of old data to prevent disk space issues
- **Secure**: Environment-based configuration for all sensitive credentials
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## 📋 Prerequisites

- Python 3.12.2 (recommended) or Python 3.13+
- 4GB+ RAM recommended
- 2GB+ free disk space for vector stores

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vivum-backend.git
   cd vivum-backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   
   Required environment variables:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase service role key
   - `TOGETHER_API_KEY`: Your Together AI API key

## 🏃‍♂️ Running the Application

```bash
python main.py
```

The server will start on `http://localhost:8000`

## 📚 API Documentation

### Core Endpoints

#### 1. Fetch Topic Data
```http
POST /fetch-topic-data
Content-Type: application/json

{
  "topic": "diabetes type 2 treatment",
  "max_results": 20,
  "filters": {
    "publication_date": "5_years",
    "article_types": ["clinical_trial", "meta_analysis"]
  }
}
```

**Response:**
```json
{
  "topic_id": "uuid-here",
  "message": "Started fetching data...",
  "status": "processing"
}
```

#### 2. Query Articles
```http
POST /query
Content-Type: application/json

{
  "topic_id": "uuid-here",
  "query": "What are the latest treatments?",
  "conversation_id": "optional-conversation-id"
}
```

#### 3. Check Topic Status
```http
GET /topic/{topic_id}/status
```

#### 4. Get Topic Articles
```http
GET /topic/{topic_id}/articles?limit=10&offset=0
```

#### 5. Cleanup Topic Data
```http
DELETE /topic/{topic_id}/cleanup
```

### Advanced Search Features

#### Multi-topic Boolean Search
```json
{
  "topics": ["diabetes", "insulin resistance", "metabolic syndrome"],
  "operator": "AND",
  "max_results": 50
}
```

#### Available Filters
- **Publication Date**: `1_year`, `5_years`, `10_years`, `custom`
- **Article Types**: `clinical_trial`, `meta_analysis`, `systematic_review`, etc.
- **Languages**: `english`, `spanish`, `french`, etc.
- **Species**: `humans`, `mice`, `rats`, etc.

### Monitoring Endpoints

- `GET /health` - Health check
- `GET /cleanup/status` - Resource usage statistics
- `GET /test-performance` - Performance metrics

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  FastAPI Server │────▶│  PubMed API     │     │  Supabase DB    │
│                 │     │                 │     │                 │
└────────┬────────┘     └─────────────────┘     └────────▲────────┘
         │                                                 │
         │              ┌─────────────────┐               │
         │              │                 │               │
         └─────────────▶│  FAISS Vector   │───────────────┘
                        │  Stores         │
                        │ (Topic-Specific)│
                        └─────────────────┘
```

### Key Components

1. **FastAPI**: REST API framework
2. **LangChain**: Orchestrates RAG pipeline
3. **FAISS**: Fast similarity search for embeddings
4. **Together AI**: LLM provider (Llama 3.1)
5. **Sentence Transformers**: Creates embeddings
6. **Supabase**: Stores metadata and topic information

## 🔒 Security

- All sensitive credentials stored in environment variables
- API keys never exposed in code
- Service role keys used for Supabase access
- `.env` file excluded from version control

## 🧹 Resource Management

The system automatically cleans up old data:
- Default: Topics older than 7 days are cleaned
- Cleanup runs every 24 hours
- Manual cleanup available via API

Configure via environment variables:
```env
CLEANUP_INTERVAL_HOURS=24
CLEANUP_DAYS_OLD=7
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **FAISS Index Not Found**
   - Ensure topic data fetch completed successfully
   - Check `vectorstores/{topic_id}/` directory exists

4. **Memory Issues**
   - Reduce `max_results` when fetching articles
   - Adjust batch size in `create_faiss_store_in_batches`
   - Run cleanup more frequently

## 📈 Performance Tips

- For large queries (>50 articles), expect 1-2 minutes processing time
- Batch size of 10 documents works well for most systems
- Keep `max_results` under 100 for optimal performance
- Use specific search terms for better PubMed results

