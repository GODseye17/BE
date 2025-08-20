#!/usr/bin/env python3
"""
RAG Pipeline Testing Script for Vivum RAG Backend
Tests complete RAG pipeline from query to answer with detailed timing
"""
import asyncio
import time
import logging
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipelineTester:
    """Test complete RAG pipeline with timing"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.topic_id = None
        
    async def test_topic_creation(self, topics: List[str]) -> Dict[str, Any]:
        """Test topic creation and data fetching"""
        print("📝 Testing Topic Creation...")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            # Create topic request
            topic_request = {
                "topics": topics,
                "operator": "AND",
                "filters": {
                    "max_results": 10,
                    "date_from": "2020-01-01",
                    "date_to": "2024-12-31"
                }
            }
            
            # Make request to create topic
            response = requests.post(
                f"http://localhost:{PORT}/fetch-topic-data",
                json=topic_request,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                self.topic_id = data.get("topic_id")
                
                duration = time.time() - start_time
                print(f"   ✅ Topic created successfully ({duration:.2f}s)")
                print(f"   📊 Topic ID: {self.topic_id}")
                
                return {
                    "status": "success",
                    "message": "Topic created successfully",
                    "duration": duration,
                    "topic_id": self.topic_id,
                    "data": data
                }
            else:
                duration = time.time() - start_time
                print(f"   ❌ Topic creation failed: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Topic creation failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def wait_for_topic_completion(self, topic_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """Wait for topic processing to complete"""
        print("⏳ Waiting for Topic Processing...")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            while time.time() - start_time < max_wait:
                # Check topic status
                response = requests.get(
                    f"http://localhost:{PORT}/topic-status/{topic_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    if status == "completed":
                        duration = time.time() - start_time
                        print(f"   ✅ Topic processing completed ({duration:.2f}s)")
                        return {
                            "status": "success",
                            "message": "Topic processing completed",
                            "duration": duration,
                            "data": data
                        }
                    elif status == "failed":
                        duration = time.time() - start_time
                        print(f"   ❌ Topic processing failed")
                        return {
                            "status": "error",
                            "message": "Topic processing failed",
                            "duration": duration,
                            "data": data
                        }
                    else:
                        print(f"   ⏳ Status: {status}...")
                        await asyncio.sleep(5)
                else:
                    await asyncio.sleep(5)
            
            duration = time.time() - start_time
            print(f"   ⚠️ Topic processing timeout after {max_wait}s")
            return {
                "status": "timeout",
                "message": f"Topic processing timeout after {max_wait}s",
                "duration": duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Error waiting for topic completion: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_basic_query(self, query: str) -> Dict[str, Any]:
        """Test basic RAG query"""
        print(f"🔍 Testing Basic Query: '{query}'")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            if not self.topic_id:
                return {
                    "status": "error",
                    "message": "No topic ID available",
                    "duration": time.time() - start_time
                }
            
            # Make query request
            query_request = {
                "topic_id": self.topic_id,
                "query": query,
                "conversation_id": f"test_conv_{int(time.time())}"
            }
            
            response = requests.post(
                f"http://localhost:{PORT}/query",
                json=query_request,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                duration = time.time() - start_time
                
                print(f"   ✅ Query successful ({duration:.2f}s)")
                print(f"   📝 Answer length: {len(data.get('answer', ''))} chars")
                print(f"   📊 Sources: {len(data.get('source_documents', []))} documents")
                
                return {
                    "status": "success",
                    "message": "Query successful",
                    "duration": duration,
                    "answer_length": len(data.get('answer', '')),
                    "sources_count": len(data.get('source_documents', [])),
                    "data": data
                }
            else:
                duration = time.time() - start_time
                print(f"   ❌ Query failed: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Query failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_enhanced_query(self, query: str) -> Dict[str, Any]:
        """Test enhanced RAG query with knowledge graph and multi-agent"""
        print(f"🚀 Testing Enhanced Query: '{query}'")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            if not self.topic_id:
                return {
                    "status": "error",
                    "message": "No topic ID available",
                    "duration": time.time() - start_time
                }
            
            # Make enhanced query request
            query_request = {
                "topic_id": self.topic_id,
                "query": query,
                "conversation_id": f"test_enhanced_{int(time.time())}"
            }
            
            response = requests.post(
                f"http://localhost:{PORT}/enhanced-query",
                json=query_request,
                timeout=120  # Longer timeout for enhanced queries
            )
            
            if response.status_code == 200:
                data = response.json()
                duration = time.time() - start_time
                
                print(f"   ✅ Enhanced query successful ({duration:.2f}s)")
                print(f"   📝 Answer length: {len(data.get('answer', ''))} chars")
                print(f"   📊 Sources: {data.get('source_documents', 0)} documents")
                print(f"   🤖 Multi-agent analysis: {'Yes' if data.get('multi_agent_analysis') else 'No'}")
                
                return {
                    "status": "success",
                    "message": "Enhanced query successful",
                    "duration": duration,
                    "answer_length": len(data.get('answer', '')),
                    "sources_count": data.get('source_documents', 0),
                    "has_multi_agent": bool(data.get('multi_agent_analysis')),
                    "processing_time": data.get('processing_time', 0),
                    "data": data
                }
            else:
                duration = time.time() - start_time
                print(f"   ❌ Enhanced query failed: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Enhanced query failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_knowledge_graph_building(self) -> Dict[str, Any]:
        """Test knowledge graph building"""
        print("🧠 Testing Knowledge Graph Building...")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            if not self.topic_id:
                return {
                    "status": "error",
                    "message": "No topic ID available",
                    "duration": time.time() - start_time
                }
            
            # Build knowledge graph
            response = requests.post(
                f"http://localhost:{PORT}/build-knowledge-graph/{self.topic_id}",
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                duration = time.time() - start_time
                
                graph_stats = data.get('graph_stats', {})
                print(f"   ✅ Knowledge graph built successfully ({duration:.2f}s)")
                print(f"   📊 Nodes: {graph_stats.get('nodes', 0)}")
                print(f"   🔗 Edges: {graph_stats.get('edges', 0)}")
                print(f"   📄 Articles processed: {graph_stats.get('articles_processed', 0)}")
                
                return {
                    "status": "success",
                    "message": "Knowledge graph built successfully",
                    "duration": duration,
                    "nodes": graph_stats.get('nodes', 0),
                    "edges": graph_stats.get('edges', 0),
                    "articles_processed": graph_stats.get('articles_processed', 0),
                    "data": data
                }
            else:
                duration = time.time() - start_time
                print(f"   ❌ Knowledge graph building failed: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Knowledge graph building failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics endpoint"""
        print("📊 Testing Performance Metrics...")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            response = requests.get(
                f"http://localhost:{PORT}/performance-metrics",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                duration = time.time() - start_time
                
                metrics = data.get('metrics', {})
                print(f"   ✅ Performance metrics retrieved ({duration:.2f}s)")
                print(f"   📈 Metrics available: {len(metrics)}")
                
                return {
                    "status": "success",
                    "message": "Performance metrics retrieved",
                    "duration": duration,
                    "metrics_count": len(metrics),
                    "data": data
                }
            else:
                duration = time.time() - start_time
                print(f"   ❌ Performance metrics failed: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Performance metrics failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test system health endpoint"""
        print("💚 Testing System Health...")
        start_time = time.time()
        
        try:
            import requests
            from config.settings import PORT
            
            response = requests.get(
                f"http://localhost:{PORT}/system-health",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                duration = time.time() - start_time
                
                health = data.get('system_health', {})
                print(f"   ✅ System health retrieved ({duration:.2f}s)")
                print(f"   💻 CPU: {health.get('cpu_usage', 0):.1f}%")
                print(f"   🧠 Memory: {health.get('memory_usage', 0):.1f}%")
                print(f"   💾 Disk: {health.get('disk_usage', 0):.1f}%")
                
                return {
                    "status": "success",
                    "message": "System health retrieved",
                    "duration": duration,
                    "cpu_usage": health.get('cpu_usage', 0),
                    "memory_usage": health.get('memory_usage', 0),
                    "disk_usage": health.get('disk_usage', 0),
                    "data": data
                }
            else:
                duration = time.time() - start_time
                print(f"   ❌ System health failed: {response.status_code}")
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ System health failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "duration": duration
            }
    
    async def run_complete_pipeline_test(self, topics: List[str], queries: List[str]) -> Dict[str, Any]:
        """Run complete RAG pipeline test"""
        print("🚀 Starting Complete RAG Pipeline Test...\n")
        
        # Test 1: Topic Creation
        topic_result = await self.test_topic_creation(topics)
        self.results["topic_creation"] = topic_result
        print()
        
        if topic_result["status"] != "success":
            return self.generate_summary()
        
        # Test 2: Wait for Topic Completion
        wait_result = await self.wait_for_topic_completion(self.topic_id)
        self.results["topic_processing"] = wait_result
        print()
        
        if wait_result["status"] != "success":
            return self.generate_summary()
        
        # Test 3: Knowledge Graph Building
        kg_result = await self.test_knowledge_graph_building()
        self.results["knowledge_graph"] = kg_result
        print()
        
        # Test 4: Basic Queries
        basic_results = []
        for i, query in enumerate(queries):
            result = await self.test_basic_query(query)
            basic_results.append(result)
            self.results[f"basic_query_{i+1}"] = result
            print()
        
        # Test 5: Enhanced Queries
        enhanced_results = []
        for i, query in enumerate(queries):
            result = await self.test_enhanced_query(query)
            enhanced_results.append(result)
            self.results[f"enhanced_query_{i+1}"] = result
            print()
        
        # Test 6: Performance Metrics
        perf_result = await self.test_performance_metrics()
        self.results["performance_metrics"] = perf_result
        print()
        
        # Test 7: System Health
        health_result = await self.test_system_health()
        self.results["system_health"] = health_result
        print()
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_time = time.time() - self.start_time
        
        # Count results
        success_count = sum(1 for r in self.results.values() if r["status"] == "success")
        error_count = sum(1 for r in self.results.values() if r["status"] == "error")
        timeout_count = sum(1 for r in self.results.values() if r["status"] == "timeout")
        
        print("📋 RAG Pipeline Test Summary:")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Errors: {error_count}")
        print(f"   ⏰ Timeouts: {timeout_count}")
        print(f"   ⏱️ Total Time: {total_time:.2f}s")
        print()
        
        # Show detailed results
        for test_name, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "error" else "⏰"
            print(f"   {status_icon} {test_name}: {result['message']} ({result['duration']:.2f}s)")
        
        # Calculate performance metrics
        query_times = []
        for key, result in self.results.items():
            if "query" in key and result["status"] == "success":
                query_times.append(result["duration"])
        
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            min_query_time = min(query_times)
            max_query_time = max(query_times)
            
            print(f"\n📈 Query Performance:")
            print(f"   🕐 Average query time: {avg_query_time:.2f}s")
            print(f"   ⚡ Fastest query: {min_query_time:.2f}s")
            print(f"   🐌 Slowest query: {max_query_time:.2f}s")
        
        return {
            "total_time": total_time,
            "success_count": success_count,
            "error_count": error_count,
            "timeout_count": timeout_count,
            "results": self.results,
            "query_performance": {
                "avg_time": avg_query_time if query_times else 0,
                "min_time": min_query_time if query_times else 0,
                "max_time": max_query_time if query_times else 0
            }
        }

async def main():
    """Main function"""
    # Test configuration
    topics = ["diabetes", "treatment"]
    queries = [
        "What are the latest treatments for diabetes?",
        "What are the risk factors for diabetes?",
        "How does diet affect diabetes management?"
    ]
    
    tester = RAGPipelineTester()
    summary = await tester.run_complete_pipeline_test(topics, queries)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rag_pipeline_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    
    if summary["error_count"] == 0:
        print("\n🎉 RAG pipeline test completed successfully!")
        return True
    else:
        print(f"\n⚠️ {summary['error_count']} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
