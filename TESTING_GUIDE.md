# 🧪 Testing Guide for Vivum RAG Backend

This guide explains how to use the testing scripts to verify your RAG pipeline is working correctly.

## 📋 Available Testing Scripts

### 1. **Health Check** (`health_check.py`)
**Purpose**: Quick system status verification
**Use Case**: Daily monitoring, pre-deployment checks

### 2. **Connection Testing** (`test_connections.py`)
**Purpose**: Comprehensive connection testing for all dependencies
**Use Case**: Initial setup verification, troubleshooting

### 3. **RAG Pipeline Testing** (`test_rag_pipeline.py`)
**Purpose**: End-to-end RAG pipeline testing with timing
**Use Case**: Performance testing, validation testing

## 🚀 Quick Start

### Prerequisites
1. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Start the server**:
   ```bash
   python main.py
   ```

3. **Run tests** (in separate terminal):
   ```bash
   # Quick health check
   python health_check.py
   
   # Full connection test
   python test_connections.py
   
   # Complete RAG pipeline test
   python test_rag_pipeline.py
   ```

## 📊 Test Scripts Details

### 1. Health Check (`health_check.py`)

**What it tests**:
- ✅ Server status (running/not running)
- ✅ Supabase database connection
- ✅ Model loading status (embeddings, LLM)
- ✅ System health (CPU, memory, disk usage)

**Output example**:
```
🏥 Vivum RAG Backend Health Check

🏥 Checking Server Status...
   ✅ Server is running (0.05s)

🏥 Checking Supabase Status...
   ✅ Supabase connected (0.12s)

🏥 Checking Model Status...
   ✅ Embedding model: loaded (0.08s)
   ✅ LLM model: loaded

🏥 Checking System Health...
   ✅ CPU: 15.2% (0.10s)
   ✅ Memory: 45.8%
   ✅ Disk: 23.1%

📋 Health Check Summary:
   ✅ Healthy: 4
   ⚠️ Warnings: 0
   ❌ Issues: 0
   ⏱️ Total Time: 0.35s

🎉 All systems are healthy!
```

**Usage**:
```bash
python health_check.py
```

### 2. Connection Testing (`test_connections.py`)

**What it tests**:
- ✅ Supabase database connection
- ✅ Together AI API connection
- ✅ OpenAI API connection (optional)
- ✅ Redis connection (optional)
- ✅ Embeddings model loading
- ✅ LLM model initialization
- ✅ PubMed API connection
- ✅ Vector store functionality

**Output example**:
```
🚀 Starting Connection Tests...

🔗 Testing Supabase Connection...
   ✅ Supabase connected successfully (0.15s)

🔗 Testing Together AI Connection...
   ✅ Together AI connected successfully (1.23s)

🔗 Testing OpenAI Connection...
   ⚠️ OpenAI API key not configured (optional)

🔗 Testing Redis Connection...
   ⚠️ Redis connection failed: Connection refused

🔗 Testing Embeddings Model...
   ✅ Embeddings model loaded successfully (2.45s)

🔗 Testing LLM Model...
   ✅ LLM model initialized successfully (1.87s)

🔗 Testing PubMed API...
   ✅ PubMed API connected successfully (0.89s)

🔗 Testing Vector Store...
   ✅ Vector store manager initialized successfully (0.12s)

📋 Connection Test Summary:
   ✅ Successful: 6
   ❌ Errors: 0
   ⚠️ Warnings: 2
   ⏱️ Total Time: 6.71s

🎉 All critical connections are working!
```

**Usage**:
```bash
python test_connections.py
```

### 3. RAG Pipeline Testing (`test_rag_pipeline.py`)

**What it tests**:
- ✅ Topic creation and data fetching
- ✅ Topic processing completion
- ✅ Knowledge graph building
- ✅ Basic RAG queries (3 test queries)
- ✅ Enhanced RAG queries (3 test queries)
- ✅ Performance metrics
- ✅ System health

**Output example**:
```
🚀 Starting Complete RAG Pipeline Test...

📝 Testing Topic Creation...
   ✅ Topic created successfully (2.34s)
   📊 Topic ID: 12345678-1234-1234-1234-123456789abc

⏳ Waiting for Topic Processing...
   ✅ Topic processing completed (45.67s)

🧠 Testing Knowledge Graph Building...
   ✅ Knowledge graph built successfully (12.34s)
   📊 Nodes: 156
   🔗 Edges: 234
   📄 Articles processed: 10

🔍 Testing Basic Query: 'What are the latest treatments for diabetes?'
   ✅ Query successful (3.45s)
   📝 Answer length: 1247 chars
   📊 Sources: 5 documents

🚀 Testing Enhanced Query: 'What are the latest treatments for diabetes?'
   ✅ Enhanced query successful (8.90s)
   📝 Answer length: 2156 chars
   📊 Sources: 5 documents
   🤖 Multi-agent analysis: Yes

📊 Testing Performance Metrics...
   ✅ Performance metrics retrieved (0.12s)
   📈 Metrics available: 8

💚 Testing System Health...
   ✅ System health retrieved (0.08s)
   💻 CPU: 23.4%
   🧠 Memory: 67.8%
   💾 Disk: 34.2%

📋 RAG Pipeline Test Summary:
   ✅ Successful: 11
   ❌ Errors: 0
   ⏰ Timeouts: 0
   ⏱️ Total Time: 72.90s

📈 Query Performance:
   🕐 Average query time: 6.18s
   ⚡ Fastest query: 3.45s
   🐌 Slowest query: 8.90s

💾 Results saved to: rag_pipeline_test_results_20241201_143022.json

🎉 RAG pipeline test completed successfully!
```

**Usage**:
```bash
python test_rag_pipeline.py
```

## 📈 Understanding Test Results

### Status Codes
- **✅ Success**: Component is working correctly
- **⚠️ Warning**: Component has issues but is not critical
- **❌ Error**: Critical issue that needs attention

### Performance Metrics
- **Query Time**: Time from request to response
- **Processing Time**: Time for multi-agent analysis
- **System Resources**: CPU, memory, disk usage

### Test Results Files
- **RAG Pipeline Test**: Results are saved to `rag_pipeline_test_results_YYYYMMDD_HHMMSS.json`
- **Contains**: Detailed timing, performance metrics, and test results

## 🔧 Troubleshooting

### Common Issues

#### 1. Server Not Running
```bash
# Start the server first
python main.py
```

#### 2. Missing Environment Variables
```bash
# Check your .env file has all required variables
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
TOGETHER_API_KEY=your_key
```

#### 3. Connection Timeouts
```bash
# Increase timeout in test scripts if needed
# Default: 60s for queries, 120s for enhanced queries
```

#### 4. Model Loading Issues
```bash
# Check if models are downloaded
# First run may take longer to download models
```

### Debug Mode
Add debug logging to any test script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Benchmarks

### Expected Performance (After Optimization)
- **Health Check**: < 1 second
- **Connection Test**: < 10 seconds
- **Basic Query**: 2-5 seconds
- **Enhanced Query**: 5-15 seconds
- **Knowledge Graph Building**: 10-30 seconds

### Performance Targets
- **Response Time**: < 5 seconds for basic queries
- **Memory Usage**: < 2GB total
- **CPU Usage**: < 80% during queries
- **Error Rate**: < 1%

## 🎯 Testing Scenarios

### 1. **Daily Health Check**
```bash
python health_check.py
```

### 2. **Pre-Deployment Testing**
```bash
python test_connections.py
python test_rag_pipeline.py
```

### 3. **Performance Testing**
```bash
# Run multiple times to get average performance
for i in {1..5}; do
    python test_rag_pipeline.py
    sleep 30
done
```

### 4. **Load Testing**
```bash
# Run multiple instances simultaneously
python test_rag_pipeline.py &
python test_rag_pipeline.py &
python test_rag_pipeline.py &
wait
```

## 📝 Customizing Tests

### Modify Test Queries
Edit `test_rag_pipeline.py`:
```python
# Change test topics
topics = ["your_topic_1", "your_topic_2"]

# Change test queries
queries = [
    "Your custom query 1?",
    "Your custom query 2?",
    "Your custom query 3?"
]
```

### Modify Timeouts
Edit timeout values in test scripts:
```python
timeout=60  # Increase for slower systems
```

### Add Custom Tests
Add new test methods to any test class:
```python
async def test_custom_functionality(self):
    # Your custom test logic
    pass
```

## 🚨 Emergency Testing

### Quick System Check
```bash
# Fastest way to check if system is working
python health_check.py
```

### Critical Path Testing
```bash
# Test only essential functionality
python test_connections.py
```

### Full System Validation
```bash
# Complete end-to-end testing
python test_rag_pipeline.py
```

## 📞 Support

If tests fail:
1. **Check logs**: Look for error messages
2. **Verify environment**: Ensure all variables are set
3. **Check dependencies**: Ensure all packages are installed
4. **Restart server**: Sometimes a restart fixes issues
5. **Check resources**: Ensure sufficient CPU/memory

---

**Happy Testing!** 🧪✨
