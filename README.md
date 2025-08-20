# Vivum Backend - World-Class RAG Pipeline

A cutting-edge research assistant backend that provides intelligent literature analysis using world-class RAG (Retrieval-Augmented Generation) techniques with knowledge graph integration, multi-agent reasoning, comprehensive relevance scoring, query enhancement, and feedback-driven optimization.

## 🚀 Key Features

### Core RAG Pipeline
- **PubMed Integration**: Direct access to PubMed's vast medical literature database
- **Multi-topic Search**: Support for complex boolean queries across multiple topics
- **Advanced Filtering**: Comprehensive filtering by publication date, article type, language, species, and more
- **Vector Storage**: FAISS-based vector storage for efficient similarity search

### 🧠 Knowledge Graph Integration
- **Medical Entity Extraction**: Advanced NER for diseases, drugs, genes, symptoms, procedures, organs
- **Relationship Extraction**: Automatic extraction of medical relationships (treats, causes, inhibits, etc.)
- **Graph-Based Retrieval**: Enhanced document retrieval using knowledge graph centrality
- **Entity Relationship Analysis**: Understanding connections between medical concepts

### 🤖 Multi-Agent System
- **Research Agent**: Literature analysis, evidence synthesis, research methodology evaluation
- **Clinical Agent**: Clinical implications, patient care recommendations, safety assessment
- **Statistical Agent**: Statistical analysis, evidence grading (GRADE), meta-analysis
- **Critic Agent**: Quality assurance, fact-checking, response validation (OpenAI-based)

### 🎯 Enhanced Relevance System
- **Comprehensive Scoring**: Multi-factor relevance scoring including:
  - Title exact match (30% weight)
  - Semantic similarity using sentence-transformers (25% weight)
  - Keyword coverage in title/abstract (20% weight)
  - MeSH term relevance (15% weight)
  - Recency scoring (10% weight)
- **Quality Filtering**: Automatic filtering of retracted articles, low-quality publication types, and outdated content
- **Adaptive Thresholds**: Dynamic relevance thresholds based on user feedback

### 🔍 Query Enhancement
- **Medical Acronym Expansion**: Automatic expansion of medical acronyms (MI → myocardial infarction)
- **Synonym Addition**: Intelligent addition of medical synonyms and related terms
- **Intent Detection**: Automatic detection of query intent (treatment, diagnosis, mechanism, etc.)
- **MeSH Term Integration**: Strategic addition of MeSH terms based on detected intent
- **Field Restrictions**: Automatic title boosting and field-specific searches

### 📊 Advanced Retrieval
- **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise document reranking
- **Query-Aware Retrieval**: Dynamic k-value selection based on query type:
  - Focused queries: k=10
  - Comparison queries: k=20
  - Comprehensive queries: k=30+
- **Similarity Threshold Filtering**: Only returns documents with similarity score ≥ 0.5
- **Contextual Compression**: LLM-based document compression for better relevance

### 🔄 Feedback Loop System
- **Article Feedback**: Users can rate individual articles as relevant/not relevant
- **Query Satisfaction**: Overall query satisfaction scoring (0-5 scale)
- **Adaptive Optimization**: Dynamic threshold adjustment based on user feedback
- **Pattern Analysis**: Query pattern recognition for continuous improvement
- **Performance Tracking**: Comprehensive analytics on system performance

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query Enhancer  │───▶│ PubMed Search   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Feedback Loop  │◀───│ Relevance Scorer │◀───│ Article Parser  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vector Storage  │◀───│  Reranker        │◀───│ Content Chunks  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐
│  LLM Response   │
└─────────────────┘
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vivum-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

## 🔧 Configuration

### Environment Variables
```env
# Database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# LLM Configuration
TOGETHER_API_KEY=your_together_api_key
TOGETHER_MODEL=meta-llama/Llama-2-70b-chat-hf

# OpenAI Configuration (Optional - for Critic Agent)
OPENAI_API_KEY=your_openai_api_key

# PubMed Configuration
PUBMED_EMAIL=your_email@domain.com
PUBMED_TOOL=your_tool_name

# System Configuration
MAX_CONVERSATIONS=100
CLEANUP_INTERVAL_HOURS=24
CLEANUP_DAYS_OLD=7
```

## 📚 API Endpoints

### Core Endpoints
- `POST /fetch-topic-data` - Fetch and process PubMed articles
- `POST /query` - Ask questions about fetched articles
- `GET /topic/{topic_id}/status` - Check topic processing status
- `GET /topic/{topic_id}/articles` - Get articles for a topic

### Enhanced Analysis Endpoints
- `POST /enhanced-query` - Enhanced query using knowledge graph and multi-agent system
- `POST /build-knowledge-graph/{topic_id}` - Build knowledge graph for a topic
- `GET /knowledge-graph-stats/{topic_id}` - Get knowledge graph statistics
- `GET /multi-agent-status/{topic_id}` - Get multi-agent system status
- `POST /enable-critic-agent` - Enable critic agent with OpenAI API key

### Feedback Endpoints
- `POST /feedback/article-relevance` - Rate article relevance
- `POST /feedback/query-satisfaction` - Rate overall query satisfaction
- `GET /feedback/summary` - Get feedback analytics
- `POST /feedback/reset-thresholds` - Reset relevance thresholds

### Utility Endpoints
- `POST /test-filters` - Test query filter construction
- `POST /transform-query` - Transform natural language to PubMed syntax
- `GET /health` - Health check
- `DELETE /topic/{topic_id}/cleanup` - Clean up topic data

## 🎯 Usage Examples

### 1. Fetch Articles with Enhanced Query
```python
import requests

# Fetch articles with multi-topic search
response = requests.post("http://localhost:8000/fetch-topic-data", json={
    "topics": ["diabetes", "cardiovascular disease"],
    "operator": "AND",
    "max_results": 50,
    "filters": {
        "publication_date": "5_years",
        "article_types": ["clinical_trial", "randomized_controlled_trial"],
        "languages": ["english"]
    }
})
```

### 2. Ask Questions with Reranking
```python
# Ask a question about fetched articles
response = requests.post("http://localhost:8000/query", json={
    "topic_id": "your_topic_id",
    "query": "What are the latest treatment options for diabetes?",
    "conversation_id": "optional_conversation_id"
})
```

### 3. Enhanced Analysis with Knowledge Graph and Multi-Agents
```python
# Build knowledge graph for a topic
response = requests.post("http://localhost:8000/build-knowledge-graph/your_topic_id")

# Enhanced query with multi-agent analysis
response = requests.post("http://localhost:8000/enhanced-query", json={
    "topic_id": "your_topic_id",
    "query": "What are the latest treatment options for diabetes?",
    "conversation_id": "optional_conversation_id"
})

# Get knowledge graph statistics
response = requests.get("http://localhost:8000/knowledge-graph-stats/your_topic_id")

# Enable critic agent (requires OpenAI API key)
response = requests.post("http://localhost:8000/enable-critic-agent", json={
    "topic_id": "your_topic_id",
    "openai_api_key": "your_openai_api_key"
})
```

### 4. Provide Feedback
```python
# Rate article relevance
response = requests.post("http://localhost:8000/feedback/article-relevance", json={
    "query": "diabetes treatment",
    "pmid": "12345678",
    "is_relevant": True,
    "user_score": 4.5
})

# Rate overall satisfaction
response = requests.post("http://localhost:8000/feedback/query-satisfaction", json={
    "query": "diabetes treatment",
    "satisfaction_score": 4.2,
    "feedback_text": "Very helpful response with good citations"
})
```

## 🔍 Advanced Features

### Query Enhancement Examples
- **Acronym Expansion**: "MI treatment" → "myocardial infarction treatment"
- **Synonym Addition**: "heart attack" → "myocardial infarction OR heart attack"
- **Intent Detection**: "How to treat diabetes" → Adds therapy-related MeSH terms
- **Field Boosting**: Automatic title boosting for better precision

### Relevance Scoring Components
1. **Title Exact Match**: Checks query terms in article titles
2. **Semantic Similarity**: Uses sentence-transformers for semantic matching
3. **Keyword Coverage**: Analyzes query term presence in title/abstract
4. **MeSH Relevance**: Matches query terms with MeSH vocabulary
5. **Recency Scoring**: Recent articles score higher (configurable)

### Quality Filtering
- **Retracted Articles**: Automatically filtered out
- **Low-Quality Types**: Editorials, letters, news filtered out
- **Age Filtering**: Articles older than 20 years filtered (configurable)
- **Language Filtering**: English articles only (configurable)

## 📊 Performance Optimizations

### Retrieval Optimizations
- **Batch Processing**: Parallel embedding generation
- **Smart Caching**: Conversation chain caching
- **Adaptive k-values**: Query-type specific retrieval counts
- **Similarity Thresholds**: Configurable relevance thresholds

### Memory Management
- **Automatic Cleanup**: Old conversations and files cleaned up
- **Vector Store Optimization**: Efficient FAISS indexing
- **Background Processing**: Non-blocking article fetching

## 🔧 Customization

### Relevance Scoring Weights
```python
# In pubmed/relevance_scorer.py
self.weights = {
    'title_exact_match': 0.3,    # Adjust these weights
    'semantic_similarity': 0.25,  # based on your needs
    'keyword_coverage': 0.2,
    'mesh_relevance': 0.15,
    'recency': 0.1
}
```

### Query Enhancement
```python
# In query/enhancer.py
# Add custom medical acronyms
self.medical_acronyms['custom'] = 'full_form'

# Add custom synonyms
self.medical_synonyms['custom_term'] = 'medical_term'
```

### Feedback Thresholds
```python
# In feedback/relevance_tracker.py
self.default_thresholds = {
    'min_score': 0.4,        # Minimum relevance score
    'high_relevance': 0.7,   # High relevance threshold
    'medium_relevance': 0.5, # Medium relevance threshold
    'low_relevance': 0.3     # Low relevance threshold
}
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use production-grade ASGI server (Gunicorn + Uvicorn)
- Set up proper logging and monitoring
- Configure database connection pooling
- Implement rate limiting and authentication
- Set up health checks and auto-scaling

## 📈 Monitoring and Analytics

### Feedback Analytics
- Query success rates
- User satisfaction scores
- Article relevance patterns
- System performance metrics

### Performance Metrics
- Query response times
- Embedding generation speed
- Memory usage patterns
- Database query performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API examples

---

**Vivum Backend** - Making medical research accessible and intelligent through advanced AI-powered literature analysis.

