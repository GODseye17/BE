# 🧪 Comprehensive Testing Guide for Vivum RAG Backend

## 📋 **Overview**
This guide provides comprehensive documentation for all testing scripts in the Vivum RAG Backend. Each test script is designed to validate specific components and ensure the system is working correctly.

---

## 🚀 **Test Scripts Overview**

### **1. Quick Health Check** (`health_check.py`)
**Purpose**: Quick verification of system status and core dependencies
**Usage**: `python health_check.py`
**Duration**: 30-60 seconds
**Best for**: Daily system verification and troubleshooting

**What it tests**:
- ✅ Server status and availability
- ✅ Supabase database connectivity
- ✅ Model loading status (embeddings, LLM)
- ✅ System health metrics (CPU, memory, disk usage)

**Expected Output**:
```
🚀 Health Check for Vivum RAG Backend
=====================================

🔍 Checking Server Status...
   ✅ Server is running on http://localhost:8000

🔍 Checking Supabase Connection...
   ✅ Supabase connected successfully

🔍 Checking Model Status...
   ✅ Embedding model loaded
   ✅ LLM model loaded

🔍 Checking System Health...
   ✅ CPU Usage: 15.2%
   ✅ Memory Usage: 45.8%
   ✅ Disk Usage: 23.1%

🎉 All health checks passed!
```

### **2. Comprehensive Connection Testing** (`test_connections.py`)
**Purpose**: Detailed testing of all external and internal connections
**Usage**: `python test_connections.py`
**Duration**: 2-3 minutes
**Best for**: Initial setup verification and debugging connection issues

**What it tests**:
- ✅ Supabase database connection and authentication
- ✅ Together AI API connectivity and model access
- ✅ OpenAI API connectivity (if configured)
- ✅ Redis cache connection and functionality
- ✅ Embeddings model loading and inference
- ✅ LLM model loading and basic generation
- ✅ PubMed API connectivity and search functionality
- ✅ Vector store initialization and basic operations

**Expected Output**:
```
🚀 Connection Testing for Vivum RAG Backend
==========================================

🔗 Testing Supabase Connection...
   ✅ Supabase connected successfully (0.85s)

🔗 Testing Together AI Connection...
   ✅ Together AI connected successfully (1.23s)

🔗 Testing OpenAI Connection...
   ⚠️ OpenAI not configured (skipped)

🔗 Testing Redis Connection...
   ✅ Redis connected successfully (0.12s)

🔗 Testing Embeddings Model...
   ✅ Embeddings model loaded successfully (2.45s)

🔗 Testing LLM Model...
   ✅ LLM model loaded successfully (3.12s)

🔗 Testing PubMed API...
   ✅ PubMed API connected successfully (0.67s)

🔗 Testing Vector Store...
   ✅ Vector store initialized successfully (1.89s)

📊 Connection Test Results
=========================
Total Tests: 8
Passed: 7
Failed: 0
Warnings: 1
Success Rate: 87.5%
Total Duration: 10.33s
```

### **3. End-to-End RAG Pipeline Testing** (`test_rag_pipeline.py`)
**Purpose**: Complete RAG pipeline testing with detailed timing analysis
**Usage**: `python test_rag_pipeline.py`
**Duration**: 5-10 minutes
**Best for**: Performance benchmarking and pipeline validation

**What it tests**:
- ✅ Complete topic creation and data fetching workflow
- ✅ Knowledge graph building and entity extraction
- ✅ Basic query processing and response generation
- ✅ Enhanced query processing with multi-agent analysis
- ✅ Performance metrics collection and analysis
- ✅ System health monitoring during pipeline execution
- ✅ Detailed timing for each pipeline component

**Expected Output**:
```
🚀 RAG Pipeline Testing for Vivum RAG Backend
=============================================

📝 Creating Test Topic...
   ✅ Topic created successfully (1.23s)

⏳ Waiting for Topic Completion...
   ✅ Topic processing completed (45.67s)

🧠 Building Knowledge Graph...
   ✅ Knowledge graph built (12.34s)
   ✅ Nodes: 156, Edges: 234

🔍 Testing Basic Queries...
   ✅ Query 1: "diabetes treatment" (2.45s)
   ✅ Query 2: "heart disease prevention" (2.12s)
   ✅ Query 3: "cancer immunotherapy" (2.78s)

🤖 Testing Enhanced Queries...
   ✅ Enhanced query 1: "diabetes treatment" (8.90s)
   ✅ Enhanced query 2: "heart disease prevention" (7.65s)

📊 Performance Metrics...
   ✅ Average response time: 2.45s
   ✅ Memory usage: 1.2GB
   ✅ CPU usage: 25.3%

🏥 System Health...
   ✅ All systems healthy

📄 Results saved to: test_results/rag_pipeline_test_20241201_143022.json
```

---

## 🔧 **Specialized Microservice Tests**

### **4. Redis Caching Test** (`test_redis_caching.py`)
**Purpose**: Test Redis caching functionality and performance improvements
**Usage**: `python test_redis_caching.py`
**Duration**: 2-3 minutes
**Requirements**: Redis server running
**Best for**: Validating caching performance and integration

**What it tests**:
- ✅ Redis connection and basic operations
- ✅ Cache performance improvements
- ✅ Cache integration with API endpoints
- ✅ Cache stress testing under high load
- ✅ Cache hit rates and response time improvements

**Expected Output**:
```
🚀 Redis Caching Test Suite
===========================

🔗 Testing Redis Connection...
   ✅ Redis caching working perfectly (0.15s)

⚡ Testing Cache Performance...
   ✅ Cache Performance Test Complete (1.23s)
      Cache Hit Rate: 85.7%
      Time Saved: 12.45s
      Performance Improvement: 78.3%

🔧 Testing Cache Integration...
   ✅ Cache Integration Test Complete (0.67s)
      Fast Responses: 3/3
      Success Rate: 100.0%

💪 Testing Cache Stress...
   ✅ Cache Stress Test Complete (2.34s)
      Concurrent Operations: 50
      Successful Operations: 50
      Success Rate: 100.0%

📊 Redis Caching Test Results
============================
Total Tests: 4
Passed: 4
Failed: 0
Success Rate: 100.0%
Total Duration: 4.39s
```

### **5. Feedback Loop Test** (`test_feedback_loop.py`)
**Purpose**: Test feedback collection, relevance tracking, and adaptive optimization
**Usage**: `python test_feedback_loop.py`
**Duration**: 2-3 minutes
**Best for**: Validating feedback system and adaptive learning

**What it tests**:
- ✅ Feedback tracker initialization
- ✅ Article relevance feedback collection
- ✅ Query satisfaction tracking
- ✅ Dynamic threshold adjustment
- ✅ Feedback API endpoints
- ✅ Feedback data persistence

**Expected Output**:
```
🚀 Feedback Loop Test Suite
===========================

🔧 Testing Feedback Tracker Initialization...
   ✅ Feedback tracker initialized successfully (0.12s)

📝 Testing Article Feedback Collection...
   ✅ Article Feedback Collection Test Complete (0.89s)
      Successful Feedbacks: 3/3
      Success Rate: 100.0%

😊 Testing Query Satisfaction Tracking...
   ✅ Query Satisfaction Tracking Test Complete (0.67s)
      Successful Satisfactions: 3/3
      Success Rate: 100.0%

⚖️ Testing Threshold Adjustment...
   ✅ Threshold Adjustment Test Complete (1.23s)
      Thresholds Changed: True
      Adjustment Triggered: True

🌐 Testing Feedback API Endpoints...
   ✅ Feedback API Endpoints Test Complete (0.45s)
      Successful Endpoints: 4/4
      Success Rate: 100.0%

💾 Testing Feedback Persistence...
   ✅ Feedback Persistence Test Complete (0.34s)
      Data Persisted: True
      Feedback Count: 3

📊 Feedback Loop Test Results
============================
Total Tests: 6
Passed: 6
Failed: 0
Success Rate: 100.0%
Total Duration: 3.70s
```

### **6. Knowledge Graph Test** (`test_knowledge_graph.py`)
**Purpose**: Test entity extraction, relationship building, and graph-based retrieval
**Usage**: `python test_knowledge_graph.py`
**Duration**: 3-4 minutes
**Best for**: Validating knowledge graph functionality and entity extraction

**What it tests**:
- ✅ Medical entity extraction from text
- ✅ Relationship extraction between entities
- ✅ Knowledge graph construction
- ✅ Graph-based document retrieval
- ✅ Entity relationship analysis
- ✅ Knowledge graph API endpoints

**Expected Output**:
```
🚀 Knowledge Graph Test Suite
=============================

🔍 Testing Entity Extraction...
   ✅ Entity Extraction Test Complete (1.45s)
      Total Entities Extracted: 23
      Entity Types: {'disease': 8, 'drug': 6, 'symptom': 5, 'procedure': 4}
      Extraction Successful: True

🔗 Testing Relationship Extraction...
   ✅ Relationship Extraction Test Complete (1.23s)
      Total Relationships Extracted: 15
      Relationship Types: {'treats': 6, 'prevents': 4, 'causes': 3, 'regulates': 2}
      Extraction Successful: True

🧠 Testing Knowledge Graph Building...
   ✅ Knowledge Graph Building Test Complete (2.34s)
      Nodes: 45, Edges: 67
      Graph Built Successfully: True

🔍 Testing Graph-Based Retrieval...
   ✅ Graph-Based Retrieval Test Complete (1.67s)
      Successful Retrievals: 4/4
      Total Documents Retrieved: 12
      Success Rate: 100.0%

🔗 Testing Entity Relationships...
   ✅ Entity Relationships Test Complete (1.12s)
      Successful Analyses: 4/4
      Total Relationships Found: 18
      Success Rate: 100.0%

🌐 Testing Knowledge Graph API...
   ✅ Knowledge Graph API Test Complete (0.89s)
      Successful Endpoints: 3/3
      Success Rate: 100.0%

📊 Knowledge Graph Test Results
==============================
Total Tests: 6
Passed: 6
Failed: 0
Success Rate: 100.0%
Total Duration: 8.70s
```

### **7. Multi-Agent System Test** (`test_multi_agent.py`)
**Purpose**: Test research, clinical, statistical, and critic agents
**Usage**: `python test_multi_agent.py`
**Duration**: 4-5 minutes
**Requirements**: Together AI API key, OpenAI API key (optional)
**Best for**: Validating multi-agent analysis and coordination

**What it tests**:
- ✅ Research agent literature analysis
- ✅ Clinical agent clinical implications
- ✅ Statistical agent evidence evaluation
- ✅ Critic agent response validation (if OpenAI configured)
- ✅ Multi-agent coordinator orchestration
- ✅ Multi-agent API endpoints

**Expected Output**:
```
🚀 Multi-Agent System Test Suite
================================

🔬 Testing Research Agent...
   ✅ Research Agent Test Complete (2.34s)
      Analysis Successful: True
      Insights Count: 5
      Confidence: 0.85

🏥 Testing Clinical Agent...
   ✅ Clinical Agent Test Complete (2.12s)
      Assessment Successful: True
      Insights Count: 4
      Confidence: 0.78

📊 Testing Statistical Agent...
   ✅ Statistical Agent Test Complete (1.89s)
      Evaluation Successful: True
      Insights Count: 3
      Confidence: 0.92

🎯 Testing Critic Agent...
   ⚠️ OpenAI API key not configured, skipping critic agent test

🤖 Testing Multi-Agent Coordinator...
   ✅ Multi-Agent Coordinator Test Complete (3.45s)
      Coordination Successful: True
      Overall Confidence: 0.82
      Agents Used: 3

🌐 Testing Multi-Agent API Endpoints...
   ✅ Multi-Agent API Endpoints Test Complete (0.67s)
      Successful Endpoints: 3/3
      Success Rate: 100.0%

📊 Multi-Agent System Test Results
=================================
Total Tests: 6
Passed: 5
Failed: 0
Warnings: 1
Success Rate: 83.3%
Total Duration: 10.47s
```

---

## 📊 **Test Results and Analysis**

### **Result Files**
All test scripts save detailed results to JSON files in the `test_results/` directory:

```
test_results/
├── health_check_20241201_143022.json
├── connections_test_20241201_143022.json
├── rag_pipeline_test_20241201_143022.json
├── redis_caching_test_20241201_143022.json
├── feedback_loop_test_20241201_143022.json
├── knowledge_graph_test_20241201_143022.json
└── multi_agent_test_20241201_143022.json
```

### **Result Format**
Each result file contains:
```json
{
  "test_type": "test_name",
  "timestamp": "20241201_143022",
  "overall_results": {
    "total_tests": 6,
    "passed_tests": 5,
    "failed_tests": 1,
    "success_rate": 83.3,
    "total_duration": 45.67
  },
  "detailed_results": {
    "test_name": {
      "status": "success",
      "message": "Test completed successfully",
      "duration": 2.34,
      "data": {
        "metric1": "value1",
        "metric2": "value2"
      }
    }
  }
}
```

---

## 🛠️ **Troubleshooting Guide**

### **Common Issues and Solutions**

#### **1. Redis Connection Failed**
**Error**: `Redis connection failed`
**Solution**: 
```bash
# Install and start Redis
brew install redis
brew services start redis

# Test Redis connection
redis-cli ping
```

#### **2. OpenAI API Key Not Configured**
**Error**: `OpenAI API key not configured`
**Solution**:
```bash
# Add to .env file
OPENAI_API_KEY=your_openai_api_key_here
```

#### **3. Server Not Running**
**Error**: `Server is not running`
**Solution**:
```bash
# Start the server
python main.py

# Or with uvicorn
uvicorn main:app --reload
```

#### **4. Model Loading Failed**
**Error**: `Model loading failed`
**Solution**:
```bash
# Check environment variables
echo $TOGETHER_API_KEY

# Reinstall dependencies
pip install -r requirements.txt
```

#### **5. Database Connection Failed**
**Error**: `Supabase connection failed`
**Solution**:
```bash
# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_KEY

# Test connection manually
python -c "from supabase import create_client; print('Connected')"
```

---

## 📈 **Performance Benchmarks**

### **Expected Performance Metrics**

| **Test Type** | **Duration** | **Success Rate** | **Performance Target** |
|---------------|--------------|------------------|------------------------|
| Health Check | 30-60s | >95% | <1s per check |
| Connection Test | 2-3min | >90% | <5s per connection |
| RAG Pipeline | 5-10min | >85% | <3s per query |
| Redis Caching | 2-3min | >95% | 80% performance improvement |
| Feedback Loop | 2-3min | >90% | <1s per feedback |
| Knowledge Graph | 3-4min | >85% | <2s per entity |
| Multi-Agent | 4-5min | >80% | <10s per analysis |

### **Performance Targets**

#### **Response Times**
- **Health Check**: <1 second per component
- **Connection Test**: <5 seconds per connection
- **RAG Pipeline**: <3 seconds per query
- **Redis Caching**: <0.1 seconds for cache hits
- **Feedback Loop**: <1 second per feedback
- **Knowledge Graph**: <2 seconds per entity extraction
- **Multi-Agent**: <10 seconds per analysis

#### **Success Rates**
- **Critical Systems**: >95% success rate
- **Core Features**: >90% success rate
- **Advanced Features**: >80% success rate

---

## 🎯 **Testing Strategy**

### **Daily Testing**
```bash
# Quick health check
python health_check.py
```

### **Weekly Testing**
```bash
# Full connection test
python test_connections.py

# RAG pipeline test
python test_rag_pipeline.py
```

### **Before Deployment**
```bash
# Run all tests
python health_check.py
python test_connections.py
python test_rag_pipeline.py
python test_redis_caching.py
python test_feedback_loop.py
python test_knowledge_graph.py
python test_multi_agent.py
```

### **Performance Testing**
```bash
# Test specific components
python test_redis_caching.py  # Cache performance
python test_knowledge_graph.py  # Entity extraction
python test_multi_agent.py  # Agent coordination
```

---

## 📞 **Support and Resources**

### **Documentation**
- **Redis**: https://redis.io/documentation
- **OpenAI API**: https://platform.openai.com/docs
- **Together AI**: https://docs.together.ai
- **Supabase**: https://supabase.com/docs

### **Monitoring Endpoints**
- **Health Check**: `GET /health`
- **Performance Metrics**: `GET /performance-metrics`
- **System Health**: `GET /system-health`

### **Logs and Debugging**
```bash
# Check application logs
tail -f logs/app.log

# Check test results
ls -la test_results/

# Analyze test performance
python -c "import json; data=json.load(open('test_results/latest.json')); print(f'Success Rate: {data[\"overall_results\"][\"success_rate\"]}%')"
```

---

## 🎉 **Success Criteria**

### **All Tests Passing**
- ✅ Health check: All components healthy
- ✅ Connection test: All external services connected
- ✅ RAG pipeline: End-to-end workflow working
- ✅ Redis caching: Performance improvements achieved
- ✅ Feedback loop: Adaptive learning working
- ✅ Knowledge graph: Entity extraction and retrieval working
- ✅ Multi-agent: Coordinated analysis working

### **Performance Targets Met**
- ✅ Response times within acceptable ranges
- ✅ Success rates above minimum thresholds
- ✅ Resource usage within limits
- ✅ Error rates below acceptable levels

### **System Ready for Production**
- ✅ All critical systems tested and validated
- ✅ Performance benchmarks achieved
- ✅ Error handling and recovery tested
- ✅ Monitoring and alerting configured

---

**Happy Testing!** 🧪✨
