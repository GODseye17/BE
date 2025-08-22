# 🚀 GraphRAG Pipeline Test Script

This directory contains test scripts to demonstrate the complete GraphRAG pipeline with timing for each step.

## 📁 Files

- `simple_graphrag_test.py` - **Recommended**: Simplified test script with better error handling
- `test_graphrag_pipeline.py` - Full-featured test script with comprehensive functionality
- `TEST_SCRIPT_README.md` - This documentation file

## 🎯 What the Test Script Does

The test script demonstrates the complete GraphRAG pipeline:

1. **System Initialization** - Loads models and connects to databases
2. **Article Retrieval** - Fetches research articles from PubMed
3. **Topic Processing** - Processes and stores articles with embeddings
4. **Query Processing** - Answers questions using the GraphRAG pipeline
5. **Timing Analysis** - Shows detailed timing for each step

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (create a `.env` file):
   ```bash
   # Required for full functionality
   TOGETHER_API_KEY=your_together_api_key_here
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_KEY=your_supabase_key_here
   
   # Optional for enhanced features
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Get API Keys**:
   - **Together AI**: https://together.ai/ (for LLM functionality)
   - **Supabase**: https://supabase.com/ (for database)
   - **OpenAI**: https://platform.openai.com/ (optional, for critic agent)

### Running the Test

```bash
# Run the simple test script (recommended)
python simple_graphrag_test.py

# Or run the full-featured test script
python test_graphrag_pipeline.py
```

## 📋 Test Flow

### Step 1: System Initialization
```
🔄 Starting: System Initialization
🔍 Checking environment variables...
🔗 Connecting to Supabase...
✅ Supabase connection successful
🧠 Loading embedding model...
✅ Embedding model loaded: sentence-transformers/multi-qa-mpnet-base-dot-v1
🤖 Loading LLM...
✅ LLM loaded successfully
✅ System initialized successfully
✅ Completed: System Initialization (5.23s)
```

### Step 2: Article Retrieval
```
🔄 Starting: Article Retrieval
📚 Fetching articles for topic: 'cancer immunotherapy'
📊 Max results: 50
📋 Topic ID: abc123-def456-ghi789
✅ Completed: Article Retrieval (12.45s)
```

### Step 3: Topic Processing
```
🔄 Starting: Topic Processing
⏳ Processing... (5s elapsed)
⏳ Processing... (10s elapsed)
⏳ Processing... (15s elapsed)
✅ Topic processing completed
✅ Completed: Topic Processing (18.67s)
```

### Step 4: Query Processing
```
🔄 Starting: Query Processing
🤔 Processing question: 'What are the latest advances in cancer immunotherapy?'
✅ Completed: Query Processing (3.21s)
```

### Step 5: Final Metrics
```
📊 PIPELINE TIMING SUMMARY
============================================================
⏱️  System Initialization    :     5.23s
⏱️  Article Retrieval        :    12.45s
⏱️  Topic Processing         :    18.67s
⏱️  Query Processing         :     3.21s
------------------------------------------------------------
⏱️  TOTAL TIME               :    39.56s
📈 STEPS COMPLETED           :        4
============================================================
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Required | Purpose | Default |
|----------|----------|---------|---------|
| `TOGETHER_API_KEY` | Yes | LLM functionality | None |
| `SUPABASE_URL` | Yes | Database connection | None |
| `SUPABASE_KEY` | Yes | Database authentication | None |
| `OPENAI_API_KEY` | No | Critic agent | None |
| `EMBEDDING_MODEL` | No | Embedding model | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| `LLM_MODEL` | No | LLM model | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |

### Test Parameters

- **Max Articles**: Default 50, can be customized during test
- **Date Range**: Default 2020-present
- **Article Types**: Research articles and reviews
- **Languages**: English only

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **API Key Errors**:
   ```bash
   # Check your .env file
   cat .env
   
   # Or set environment variables
   export TOGETHER_API_KEY="your_key_here"
   ```

3. **Supabase Connection Issues**:
   ```bash
   # Test Supabase connection
   curl -X GET "https://your-project.supabase.co/rest/v1/" \
     -H "apikey: your_supabase_key"
   ```

4. **Memory Issues**:
   ```bash
   # Reduce max articles
   # Use smaller embedding model
   # Enable memory monitoring
   ```

### Error Messages

| Error | Solution |
|-------|----------|
| `TOGETHER_API_KEY must be set` | Get API key from https://together.ai/ |
| `Supabase connection failed` | Check URL and key in .env file |
| `PubMed fetcher not available` | Install required dependencies |
| `Topic processing timeout` | Increase timeout or reduce article count |

## 📊 Performance Metrics

The test script provides detailed timing for:

- **System Initialization**: Model loading and connection setup
- **Article Retrieval**: PubMed API calls and data fetching
- **Topic Processing**: Embedding generation and storage
- **Query Processing**: GraphRAG pipeline execution
- **Total Time**: Complete end-to-end processing

### Expected Performance

| Component | Expected Time | Factors |
|-----------|---------------|---------|
| System Init | 3-10s | Model size, network speed |
| Article Retrieval | 10-30s | Article count, API rate limits |
| Topic Processing | 15-60s | Article count, embedding model |
| Query Processing | 2-10s | Query complexity, LLM model |
| **Total** | **30-110s** | All factors combined |

## 🔍 Advanced Usage

### Custom Topics

Try different research topics:
- `cancer immunotherapy`
- `machine learning healthcare`
- `COVID-19 vaccines`
- `climate change mitigation`
- `quantum computing applications`

### Custom Questions

Ask various types of questions:
- **Factual**: "What are the main types of cancer immunotherapy?"
- **Comparative**: "How do different immunotherapy approaches compare?"
- **Temporal**: "What are the latest advances in the last 2 years?"
- **Analytical**: "What are the challenges in immunotherapy development?"

### Batch Testing

For automated testing, you can modify the script to:
- Test multiple topics automatically
- Generate performance reports
- Compare different configurations
- Run regression tests

## 📈 Monitoring and Logging

The test script provides:

- **Real-time progress**: Step-by-step status updates
- **Detailed timing**: Precise timing for each component
- **Error handling**: Graceful failure with informative messages
- **Logging**: Detailed logs saved to `pipeline_test.log`

### Log File

```bash
# View detailed logs
tail -f pipeline_test.log

# Search for errors
grep "ERROR" pipeline_test.log

# Search for timing information
grep "Completed" pipeline_test.log
```

## 🎯 Expected Output

A successful test run will show:

```
🚀 SIMPLE GRAPHRAG PIPELINE TEST
============================================================
This script demonstrates the GraphRAG pipeline with timing.
============================================================

📝 STEP 1: Enter a research topic
----------------------------------------
Enter a research topic (e.g., 'cancer immunotherapy'): cancer immunotherapy

📚 STEP 2: Fetching articles for 'cancer immunotherapy'
----------------------------------------
Enter max number of articles (default 50): 25

🤔 STEP 3: Ask questions about the articles
----------------------------------------
You can ask multiple questions. Type 'quit' to exit.

Enter your question: What are the main types of cancer immunotherapy?

💡 Answer:
----------------------------------------
Based on the research articles, there are several main types of cancer immunotherapy:

1. **Checkpoint Inhibitors**: These drugs block proteins that prevent immune cells from attacking cancer cells...
2. **CAR-T Cell Therapy**: This involves genetically modifying a patient's T cells...
3. **Cancer Vaccines**: These stimulate the immune system to recognize and attack cancer cells...
4. **Cytokine Therapy**: Uses proteins that help regulate immune responses...

📄 Sources: 8 documents referenced
🏷️  Entities: 15 entities extracted

📊 STEP 4: Final metrics
----------------------------------------

============================================================
📊 PIPELINE TIMING SUMMARY
============================================================
⏱️  System Initialization    :     4.56s
⏱️  Article Retrieval        :    15.23s
⏱️  Topic Processing         :    22.45s
⏱️  Query Processing         :     4.12s
------------------------------------------------------------
⏱️  TOTAL TIME               :    46.36s
📈 STEPS COMPLETED           :        4
============================================================

✅ Test completed successfully!
```

## 🤝 Contributing

To improve the test script:

1. **Add new features**: Additional pipeline components
2. **Improve error handling**: Better fallback mechanisms
3. **Enhance timing**: More granular performance metrics
4. **Add tests**: Automated test cases
5. **Update documentation**: Keep this README current

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the log file for detailed error messages
3. Verify all environment variables are set correctly
4. Ensure all dependencies are installed
5. Check the integration requirements document

---

**Happy testing! 🚀**
