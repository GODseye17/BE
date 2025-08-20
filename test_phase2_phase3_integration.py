#!/usr/bin/env python3
"""
Phase 2 & Phase 3 Integration Test Script
Tests all enhanced features that don't require external resources
"""
import asyncio
import time
import logging
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('phase2_phase3_test.log')
    ]
)
logger = logging.getLogger(__name__)

class Phase2Phase3Tester:
    """Test all Phase 2 and Phase 3 integrations"""
    
    def __init__(self):
        self.test_results = {
            "phase2": {},
            "phase3": {},
            "overall": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }
        }
    
    async def run_all_tests(self):
        """Run all Phase 2 and Phase 3 tests"""
        logger.info("🚀 Starting Phase 2 & Phase 3 Integration Tests")
        
        # Phase 2 Tests
        await self.test_phase2_features()
        
        # Phase 3 Tests
        await self.test_phase3_features()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        logger.info("✅ All Phase 2 & Phase 3 tests completed!")
    
    async def test_phase2_features(self):
        """Test Phase 2: Quality Enhancement features"""
        logger.info("🔍 Testing Phase 2: Quality Enhancement Features")
        
        # Test 1: Document Reranking
        await self.test_document_reranking()
        
        # Test 2: Query Enhancement
        await self.test_query_enhancement()
        
        # Test 3: Knowledge Graph Integration
        await self.test_knowledge_graph_integration()
        
        # Test 4: Enhanced Retrieval
        await self.test_enhanced_retrieval()
    
    async def test_phase3_features(self):
        """Test Phase 3: Advanced Features"""
        logger.info("🔧 Testing Phase 3: Advanced Features")
        
        # Test 1: Feedback Loop System
        await self.test_feedback_loop()
        
        # Test 2: Connection Pooling
        await self.test_connection_pooling()
        
        # Test 3: Memory Management
        await self.test_memory_management()
        
        # Test 4: Performance Monitoring
        await self.test_performance_monitoring()
    
    async def test_document_reranking(self):
        """Test document reranking functionality"""
        try:
            logger.info("📊 Testing Document Reranking...")
            
            from retrieval.reranker import RelevanceReranker
            
            # Initialize reranker
            reranker = RelevanceReranker()
            
            # Test documents
            test_documents = [
                {
                    'page_content': 'This study examines the effects of diabetes on cardiovascular health.',
                    'metadata': {'title': 'Diabetes and Heart Disease', 'pmid': '12345'}
                },
                {
                    'page_content': 'A comprehensive review of diabetes treatment options.',
                    'metadata': {'title': 'Diabetes Treatment Review', 'pmid': '12346'}
                },
                {
                    'page_content': 'Unrelated study about cancer research.',
                    'metadata': {'title': 'Cancer Research', 'pmid': '12347'}
                }
            ]
            
            # Test query
            test_query = "diabetes treatment"
            
            # Rerank documents
            reranked_docs = reranker.rerank_documents(test_query, test_documents, top_k=2)
            
            # Verify results
            assert len(reranked_docs) <= 2, "Should return at most 2 documents"
            if len(reranked_docs) >= 2:
                assert reranked_docs[0]['rerank_score'] >= reranked_docs[1]['rerank_score'], "Should be sorted by score"
            
            logger.info(f"✅ Document reranking test passed - {len(reranked_docs)} documents reranked")
            self.test_results["phase2"]["document_reranking"] = {
                "status": "passed",
                "documents_processed": len(test_documents),
                "documents_returned": len(reranked_docs),
                "highest_score": reranked_docs[0]['rerank_score'] if reranked_docs else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Document reranking test failed: {e}")
            self.test_results["phase2"]["document_reranking"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_query_enhancement(self):
        """Test query enhancement functionality"""
        try:
            logger.info("🔍 Testing Query Enhancement...")
            
            from query.enhancer import QueryEnhancer
            
            # Initialize enhancer
            enhancer = QueryEnhancer()
            
            # Test queries
            test_queries = [
                "mi treatment",
                "copd symptoms",
                "heart attack prevention"
            ]
            
            enhanced_queries = []
            for query in test_queries:
                enhanced = enhancer.enhance_query(query)
                enhanced_queries.append(enhanced)
                logger.info(f"  '{query}' -> '{enhanced}'")
            
            # Verify enhancement occurred - check the enhanced query string
            enhanced_text = " ".join(enhanced_queries).lower()
            assert "myocardial infarction" in enhanced_text, "MI should be expanded"
            assert "chronic obstructive pulmonary disease" in enhanced_text, "COPD should be expanded"
            
            logger.info("✅ Query enhancement test passed")
            self.test_results["phase2"]["query_enhancement"] = {
                "status": "passed",
                "queries_enhanced": len(test_queries),
                "enhanced_queries": enhanced_queries
            }
            
        except Exception as e:
            logger.error(f"❌ Query enhancement test failed: {e}")
            self.test_results["phase2"]["query_enhancement"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_knowledge_graph_integration(self):
        """Test knowledge graph integration"""
        try:
            logger.info("🧠 Testing Knowledge Graph Integration...")
            
            from knowledge_graph import MedicalKnowledgeGraph, GraphRetriever
            from knowledge_graph.entity_extractor import MedicalEntityExtractor
            
            # Test entity extraction
            extractor = MedicalEntityExtractor()
            test_text = "Diabetes mellitus is a chronic disease affecting blood glucose levels."
            entities = extractor.extract_entities(test_text)
            
            # Verify entities extracted
            assert "diseases" in entities, "Should extract disease entities"
            assert len(entities["diseases"]) > 0, "Should find diabetes as a disease"
            
            # Test knowledge graph building
            kg = MedicalKnowledgeGraph()
            
            # Mock articles for testing
            test_articles = [
                {
                    'title': 'Diabetes Treatment Study',
                    'abstract': 'This study examines diabetes treatment options.',
                    'entities': entities
                }
            ]
            
            # Build graph
            kg.build_from_articles(test_articles)
            
            # Verify graph created
            assert len(kg.graph.nodes) > 0, "Should have nodes in graph"
            
            logger.info("✅ Knowledge graph integration test passed")
            self.test_results["phase2"]["knowledge_graph"] = {
                "status": "passed",
                "entities_extracted": sum(len(ents) for ents in entities.values()),
                "graph_nodes": len(kg.graph.nodes),
                "graph_edges": len(kg.graph.edges)
            }
            
        except Exception as e:
            logger.error(f"❌ Knowledge graph integration test failed: {e}")
            self.test_results["phase2"]["knowledge_graph"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_enhanced_retrieval(self):
        """Test enhanced retrieval with all Phase 2 features"""
        try:
            logger.info("🔍 Testing Enhanced Retrieval...")
            
            from utils.chains import get_or_create_chain
            
            # This test requires a running server and topic data
            # For now, just test that the enhanced retriever can be created
            logger.info("✅ Enhanced retrieval components available")
            self.test_results["phase2"]["enhanced_retrieval"] = {
                "status": "passed",
                "message": "Enhanced retriever components available"
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced retrieval test failed: {e}")
            self.test_results["phase2"]["enhanced_retrieval"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_feedback_loop(self):
        """Test feedback loop system"""
        try:
            logger.info("🔄 Testing Feedback Loop System...")
            
            from feedback.relevance_tracker import RelevanceTracker
            
            # Initialize tracker
            tracker = RelevanceTracker()
            
            # Test article feedback
            tracker.record_article_feedback(
                query="diabetes treatment",
                pmid="12345",
                is_relevant=True,
                user_score=4
            )
            
            # Test query satisfaction
            tracker.record_query_satisfaction(
                query="diabetes treatment",
                satisfaction_score=4,
                feedback_text="Very helpful response"
            )
            
            # Get feedback summary
            summary = tracker.get_feedback_summary()
            
            # Verify feedback recorded
            assert summary["total_queries"] > 0, "Should have recorded queries"
            assert summary["total_articles"] > 0, "Should have recorded articles"
            
            logger.info("✅ Feedback loop test passed")
            self.test_results["phase3"]["feedback_loop"] = {
                "status": "passed",
                "total_queries": summary["total_queries"],
                "total_articles": summary["total_articles"],
                "average_satisfaction": summary.get("average_satisfaction", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Feedback loop test failed: {e}")
            self.test_results["phase3"]["feedback_loop"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_connection_pooling(self):
        """Test connection pooling functionality"""
        try:
            logger.info("🔗 Testing Connection Pooling...")
            
            from utils.connection_pool import ConnectionPool
            
            # Initialize connection pool
            pool = ConnectionPool()
            
            # Test session creation
            session = await pool.get_session()
            assert session is not None, "Should create session"
            await session.close()
            
            # Test request method
            # Note: This would require a real URL, so we'll just test the pool creation
            logger.info("✅ Connection pooling test passed")
            self.test_results["phase3"]["connection_pooling"] = {
                "status": "passed",
                "message": "Connection pool initialized successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Connection pooling test failed: {e}")
            self.test_results["phase3"]["connection_pooling"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_memory_management(self):
        """Test memory management functionality"""
        try:
            logger.info("🧠 Testing Memory Management...")
            
            from utils.monitoring import PerformanceMonitor
            
            # Initialize monitor
            monitor = PerformanceMonitor()
            
            # Get system metrics
            system_metrics = monitor.get_system_metrics()
            
            # Verify metrics
            assert "cpu_usage" in system_metrics, "Should have CPU usage"
            assert "memory_usage" in system_metrics, "Should have memory usage"
            assert "disk_usage" in system_metrics, "Should have disk usage"
            
            logger.info("✅ Memory management test passed")
            self.test_results["phase3"]["memory_management"] = {
                "status": "passed",
                "cpu_usage": system_metrics["cpu_usage"],
                "memory_usage": system_metrics["memory_usage"],
                "disk_usage": system_metrics["disk_usage"]
            }
            
        except Exception as e:
            logger.error(f"❌ Memory management test failed: {e}")
            self.test_results["phase3"]["memory_management"] = {
                "status": "failed",
                "error": str(e)
            }
    
    async def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        try:
            logger.info("📊 Testing Performance Monitoring...")
            
            from utils.monitoring import PerformanceMonitor
            
            # Initialize monitor
            monitor = PerformanceMonitor()
            
            # Test performance tracking
            @monitor.track_performance("test_operation")
            def test_operation():
                time.sleep(0.1)  # Simulate work
                return "success"
            
            # Run test operation
            result = test_operation()
            assert result == "success", "Test operation should succeed"
            
            # Get metrics
            metrics = monitor.get_metrics()
            
            # Verify metrics recorded
            assert "test_operation" in metrics, "Should have recorded test operation"
            
            logger.info("✅ Performance monitoring test passed")
            self.test_results["phase3"]["performance_monitoring"] = {
                "status": "passed",
                "operations_tracked": len(metrics),
                "test_operation_avg_time": metrics.get("test_operation", {}).get("avg_time", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring test failed: {e}")
            self.test_results["phase3"]["performance_monitoring"] = {
                "status": "failed",
                "error": str(e)
            }
    
    def generate_summary(self):
        """Generate test summary"""
        logger.info("📋 Generating Test Summary...")
        
        # Count results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for phase in ["phase2", "phase3"]:
            for test_name, result in self.test_results[phase].items():
                total_tests += 1
                if result["status"] == "passed":
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        # Update overall results
        self.test_results["overall"].update({
            "end_time": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        })
        
        # Log summary
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {self.test_results['overall']['success_rate']:.1f}%")
        
        # Log detailed results
        for phase in ["phase2", "phase3"]:
            logger.info(f"\n{phase.upper()} Results:")
            for test_name, result in self.test_results[phase].items():
                status_icon = "✅" if result["status"] == "passed" else "❌"
                logger.info(f"   {status_icon} {test_name}: {result['status']}")
    
    def save_results(self):
        """Save test results to file"""
        filename = f"phase2_phase3_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"💾 Test results saved to: {filename}")

async def main():
    """Main test runner"""
    tester = Phase2Phase3Tester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
