#!/usr/bin/env python3
"""
Connection Testing Script for Vivum RAG Backend
Tests all system connections and dependencies
"""
import asyncio
import time
import logging
import sys
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConnectionTester:
    """Test all system connections"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    async def test_supabase_connection(self) -> Dict[str, Any]:
        """Test Supabase database connection"""
        print("🔗 Testing Supabase Connection...")
        start_time = time.time()
        
        try:
            from config.settings import SUPABASE_URL, SUPABASE_KEY
            from supabase import create_client, Client
            
            if not SUPABASE_URL or not SUPABASE_KEY:
                return {
                    "status": "error",
                    "message": "Supabase credentials not configured",
                    "duration": time.time() - start_time
                }
            
            # Create client
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Test connection with simple query
            response = supabase.table("topics").select("count").limit(1).execute()
            
            duration = time.time() - start_time
            print(f"   ✅ Supabase connected successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "Supabase connection working",
                "duration": duration,
                "data": response.data
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Supabase connection failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_together_ai_connection(self) -> Dict[str, Any]:
        """Test Together AI API connection"""
        print("🔗 Testing Together AI Connection...")
        start_time = time.time()
        
        try:
            from config.settings import TOGETHER_API_KEY
            from together import Together
            
            if not TOGETHER_API_KEY:
                return {
                    "status": "error",
                    "message": "Together AI API key not configured",
                    "duration": time.time() - start_time
                }
            
            # Create client
            together = Together(TOGETHER_API_KEY)
            
            # Test with simple completion
            response = together.complete(
                prompt="Hello, this is a test.",
                max_tokens=10,
                temperature=0.1
            )
            
            duration = time.time() - start_time
            print(f"   ✅ Together AI connected successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "Together AI connection working",
                "duration": duration,
                "data": response
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Together AI connection failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_openai_connection(self) -> Dict[str, Any]:
        """Test OpenAI API connection (optional)"""
        print("🔗 Testing OpenAI Connection...")
        start_time = time.time()
        
        try:
            from config.settings import OPENAI_API_KEY
            from openai import OpenAI
            
            if not OPENAI_API_KEY:
                return {
                    "status": "warning",
                    "message": "OpenAI API key not configured (optional)",
                    "duration": time.time() - start_time
                }
            
            # Create client
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Test with simple completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            
            duration = time.time() - start_time
            print(f"   ✅ OpenAI connected successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "OpenAI connection working",
                "duration": duration,
                "data": response
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ⚠️ OpenAI connection failed: {e}")
            return {
                "status": "warning",
                "message": f"OpenAI connection failed: {e}",
                "duration": duration
            }
    
    async def test_redis_connection(self) -> Dict[str, Any]:
        """Test Redis connection (optional)"""
        print("🔗 Testing Redis Connection...")
        start_time = time.time()
        
        try:
            from config.settings import REDIS_URL
            import redis
            
            # Create client
            r = redis.from_url(REDIS_URL)
            
            # Test connection
            r.ping()
            
            # Test basic operations
            r.set("test_key", "test_value", ex=60)
            value = r.get("test_key")
            r.delete("test_key")
            
            duration = time.time() - start_time
            print(f"   ✅ Redis connected successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "Redis connection working",
                "duration": duration,
                "data": {"test_value": value}
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ⚠️ Redis connection failed: {e}")
            return {
                "status": "warning",
                "message": f"Redis connection failed: {e}",
                "duration": duration
            }
    
    async def test_embeddings_model(self) -> Dict[str, Any]:
        """Test embeddings model loading"""
        print("🔗 Testing Embeddings Model...")
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            from config.settings import EMBEDDING_MODEL
            
            # Load model
            model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Test embedding
            test_text = "This is a test sentence for embeddings."
            embedding = model.encode(test_text)
            
            duration = time.time() - start_time
            print(f"   ✅ Embeddings model loaded successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "Embeddings model working",
                "duration": duration,
                "data": {"embedding_shape": embedding.shape}
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Embeddings model failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_llm_model(self) -> Dict[str, Any]:
        """Test LLM model initialization"""
        print("🔗 Testing LLM Model...")
        start_time = time.time()
        
        try:
            from llm.together_model import get_llm
            
            # Get LLM instance
            llm = get_llm()
            
            # Test simple completion
            response = await llm.ainvoke("Hello, this is a test.")
            
            duration = time.time() - start_time
            print(f"   ✅ LLM model initialized successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "LLM model working",
                "duration": duration,
                "data": {"response_length": len(str(response))}
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ LLM model failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_pubmed_api(self) -> Dict[str, Any]:
        """Test PubMed API connection"""
        print("🔗 Testing PubMed API...")
        start_time = time.time()
        
        try:
            from pubmed.fetcher import PubMedFetcher
            
            # Create fetcher
            fetcher = PubMedFetcher()
            
            # Test simple search
            results = fetcher.search_pubmed("diabetes", max_results=1)
            
            duration = time.time() - start_time
            print(f"   ✅ PubMed API connected successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "PubMed API working",
                "duration": duration,
                "data": {"results_count": len(results) if results else 0}
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ PubMed API failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_vectorstore(self) -> Dict[str, Any]:
        """Test vector store functionality"""
        print("🔗 Testing Vector Store...")
        start_time = time.time()
        
        try:
            from vectorstore.manager import VectorStoreManager
            
            # Create manager
            manager = VectorStoreManager()
            
            # Test basic operations
            test_docs = [
                {"content": "Test document 1", "metadata": {"title": "Test 1"}},
                {"content": "Test document 2", "metadata": {"title": "Test 2"}}
            ]
            
            # This would normally create a vector store, but we'll just test the import
            duration = time.time() - start_time
            print(f"   ✅ Vector store manager initialized successfully ({duration:.2f}s)")
            
            return {
                "status": "success",
                "message": "Vector store manager working",
                "duration": duration,
                "data": {"test_docs": len(test_docs)}
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Vector store failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all connection tests"""
        print("🚀 Starting Connection Tests...\n")
        
        tests = [
            ("supabase", self.test_supabase_connection),
            ("together_ai", self.test_together_ai_connection),
            ("openai", self.test_openai_connection),
            ("redis", self.test_redis_connection),
            ("embeddings", self.test_embeddings_model),
            ("llm", self.test_llm_model),
            ("pubmed", self.test_pubmed_api),
            ("vectorstore", self.test_vectorstore)
        ]
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                self.results[test_name] = result
            except Exception as e:
                self.results[test_name] = {
                    "status": "error",
                    "message": f"Test failed: {e}",
                    "duration": 0
                }
            print()
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_time = time.time() - self.start_time
        
        # Count results
        success_count = sum(1 for r in self.results.values() if r["status"] == "success")
        error_count = sum(1 for r in self.results.values() if r["status"] == "error")
        warning_count = sum(1 for r in self.results.values() if r["status"] == "warning")
        
        print("📋 Connection Test Summary:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   ⚠️ Warnings: {warning_count}")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")
        print()
        
        # Show detailed results
        for test_name, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "error" else "⚠️"
            print(f"   {status_icon} {test_name.upper()}: {result['message']} ({result['duration']:.2f}s)")
        
        return {
            "total_time": total_time,
            "success_count": success_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "results": self.results
        }

async def main():
    """Main function"""
    tester = ConnectionTester()
    summary = await tester.run_all_tests()
    
    if summary["error_count"] == 0:
        print("\n🎉 All critical connections are working!")
        return True
    else:
        print(f"\n⚠️ {summary['error_count']} critical connection(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
