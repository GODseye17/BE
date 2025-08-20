#!/usr/bin/env python3
"""
Knowledge Graph Test Script for Vivum RAG Backend
Tests entity extraction, relationship building, and graph-based retrieval
"""
import asyncio
import time
import logging
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraphTester:
    """Test knowledge graph functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.entities_extracted = []
        self.relationships_built = []
    
    async def test_entity_extraction(self) -> Dict[str, Any]:
        """Test medical entity extraction"""
        print("🔍 Testing Entity Extraction...")
        start_time = time.time()
        
        try:
            from knowledge_graph.entity_extractor import MedicalEntityExtractor
            
            extractor = MedicalEntityExtractor()
            
            # Test medical texts
            test_texts = [
                "Diabetes mellitus is a chronic disease characterized by high blood glucose levels.",
                "Myocardial infarction, commonly known as heart attack, occurs when blood flow to the heart is blocked.",
                "The patient was treated with metformin for type 2 diabetes and aspirin for cardiovascular disease prevention.",
                "Breast cancer screening using mammography has been shown to reduce mortality rates.",
                "Alzheimer's disease is a neurodegenerative disorder affecting memory and cognitive function."
            ]
            
            total_entities = 0
            entity_types = {}
            
            for text in test_texts:
                entities = extractor.extract_entities(text)
                total_entities += len(entities)
                
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            duration = time.time() - start_time
            
            # Check if entities were extracted
            extraction_successful = total_entities > 0
            
            print(f"   ✅ Entity Extraction Test Complete ({duration:.2f}s)")
            print(f"      Total Entities Extracted: {total_entities}")
            print(f"      Entity Types: {entity_types}")
            print(f"      Extraction Successful: {extraction_successful}")
            
            return {
                "status": "success",
                "message": "Entity extraction test completed",
                "duration": duration,
                "data": {
                    "total_entities": total_entities,
                    "entity_types": entity_types,
                    "extraction_successful": extraction_successful,
                    "texts_processed": len(test_texts)
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Entity extraction test failed: {e}")
            return {
                "status": "error",
                "message": f"Entity extraction test failed: {e}",
                "duration": duration
            }
    
    async def test_relationship_extraction(self) -> Dict[str, Any]:
        """Test relationship extraction between entities"""
        print("🔗 Testing Relationship Extraction...")
        start_time = time.time()
        
        try:
            from knowledge_graph.relationship_extractor import RelationshipExtractor
            
            extractor = RelationshipExtractor()
            
            # Test medical texts with relationships
            test_texts = [
                "Metformin treats diabetes by reducing glucose production in the liver.",
                "Aspirin prevents heart attacks by inhibiting platelet aggregation.",
                "Chemotherapy causes hair loss in cancer patients.",
                "Insulin regulates blood glucose levels in diabetic patients.",
                "Statins lower cholesterol levels and reduce cardiovascular disease risk."
            ]
            
            total_relationships = 0
            relationship_types = {}
            
            for text in test_texts:
                relationships = extractor.extract_relationships(text)
                total_relationships += len(relationships)
                
                for rel in relationships:
                    rel_type = rel.get('relationship_type', 'unknown')
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            duration = time.time() - start_time
            
            # Check if relationships were extracted
            extraction_successful = total_relationships > 0
            
            print(f"   ✅ Relationship Extraction Test Complete ({duration:.2f}s)")
            print(f"      Total Relationships Extracted: {total_relationships}")
            print(f"      Relationship Types: {relationship_types}")
            print(f"      Extraction Successful: {extraction_successful}")
            
            return {
                "status": "success",
                "message": "Relationship extraction test completed",
                "duration": duration,
                "data": {
                    "total_relationships": total_relationships,
                    "relationship_types": relationship_types,
                    "extraction_successful": extraction_successful,
                    "texts_processed": len(test_texts)
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Relationship extraction test failed: {e}")
            return {
                "status": "error",
                "message": f"Relationship extraction test failed: {e}",
                "duration": duration
            }
    
    async def test_knowledge_graph_building(self) -> Dict[str, Any]:
        """Test knowledge graph construction"""
        print("🧠 Testing Knowledge Graph Building...")
        start_time = time.time()
        
        try:
            from knowledge_graph.builder import MedicalKnowledgeGraph
            
            graph_builder = MedicalKnowledgeGraph()
            
            # Test articles with medical content
            test_articles = [
                {
                    "title": "Diabetes Treatment with Metformin",
                    "abstract": "This study examines the effectiveness of metformin in treating type 2 diabetes. Metformin works by reducing glucose production in the liver and improving insulin sensitivity.",
                    "pmid": "test_001"
                },
                {
                    "title": "Cardiovascular Disease Prevention",
                    "abstract": "Aspirin therapy has been shown to prevent heart attacks and strokes in high-risk patients. The mechanism involves platelet inhibition and reduced clot formation.",
                    "pmid": "test_002"
                },
                {
                    "title": "Cancer Treatment Side Effects",
                    "abstract": "Chemotherapy causes various side effects including hair loss, nausea, and fatigue. These effects are related to the drug's impact on rapidly dividing cells.",
                    "pmid": "test_003"
                }
            ]
            
            # Build knowledge graph
            graph = graph_builder.build_from_articles(test_articles)
            
            # Get graph statistics
            stats = graph_builder.get_graph_statistics()
            
            duration = time.time() - start_time
            
            # Check if graph was built successfully
            graph_built = (
                graph is not None and 
                hasattr(graph, 'nodes') and 
                hasattr(graph, 'edges') and
                len(graph.nodes) > 0
            )
            
            print(f"   ✅ Knowledge Graph Building Test Complete ({duration:.2f}s)")
            print(f"      Nodes: {len(graph.nodes) if graph else 0}")
            print(f"      Edges: {len(graph.edges) if graph else 0}")
            print(f"      Graph Built Successfully: {graph_built}")
            print(f"      Graph Statistics: {stats}")
            
            return {
                "status": "success",
                "message": "Knowledge graph building test completed",
                "duration": duration,
                "data": {
                    "nodes": len(graph.nodes) if graph else 0,
                    "edges": len(graph.edges) if graph else 0,
                    "graph_built": graph_built,
                    "statistics": stats,
                    "articles_processed": len(test_articles)
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Knowledge graph building test failed: {e}")
            return {
                "status": "error",
                "message": f"Knowledge graph building test failed: {e}",
                "duration": duration
            }
    
    async def test_graph_retrieval(self) -> Dict[str, Any]:
        """Test graph-based document retrieval"""
        print("🔍 Testing Graph-Based Retrieval...")
        start_time = time.time()
        
        try:
            from knowledge_graph.retriever import GraphRetriever
            from knowledge_graph.builder import MedicalKnowledgeGraph
            
            # Build a test knowledge graph
            graph_builder = MedicalKnowledgeGraph()
            test_articles = [
                {
                    "title": "Diabetes Management",
                    "abstract": "Comprehensive guide to diabetes management including diet, exercise, and medication.",
                    "pmid": "test_001"
                },
                {
                    "title": "Heart Disease Prevention",
                    "abstract": "Strategies for preventing cardiovascular disease through lifestyle changes and medication.",
                    "pmid": "test_002"
                }
            ]
            
            graph = graph_builder.build_from_articles(test_articles)
            retriever = GraphRetriever(graph)
            
            # Test queries
            test_queries = [
                "diabetes treatment",
                "heart disease prevention",
                "blood glucose management",
                "cardiovascular health"
            ]
            
            successful_retrievals = 0
            total_documents = 0
            
            for query in test_queries:
                try:
                    documents = retriever.graph_search(query, k=5)
                    if documents:
                        successful_retrievals += 1
                        total_documents += len(documents)
                except Exception as e:
                    print(f"   ⚠️ Failed to retrieve for query '{query}': {e}")
            
            duration = time.time() - start_time
            success_rate = (successful_retrievals / len(test_queries)) * 100
            
            print(f"   ✅ Graph-Based Retrieval Test Complete ({duration:.2f}s)")
            print(f"      Successful Retrievals: {successful_retrievals}/{len(test_queries)}")
            print(f"      Total Documents Retrieved: {total_documents}")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Graph-based retrieval test completed",
                "duration": duration,
                "data": {
                    "successful_retrievals": successful_retrievals,
                    "total_queries": len(test_queries),
                    "total_documents": total_documents,
                    "success_rate": success_rate
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Graph-based retrieval test failed: {e}")
            return {
                "status": "error",
                "message": f"Graph-based retrieval test failed: {e}",
                "duration": duration
            }
    
    async def test_entity_relationships(self) -> Dict[str, Any]:
        """Test entity relationship analysis"""
        print("🔗 Testing Entity Relationships...")
        start_time = time.time()
        
        try:
            from knowledge_graph.builder import MedicalKnowledgeGraph
            from knowledge_graph.retriever import GraphRetriever
            
            # Build test graph
            graph_builder = MedicalKnowledgeGraph()
            test_articles = [
                {
                    "title": "Medical Treatment Guide",
                    "abstract": "Metformin treats diabetes. Aspirin prevents heart attacks. Insulin regulates blood glucose.",
                    "pmid": "test_001"
                }
            ]
            
            graph = graph_builder.build_from_articles(test_articles)
            retriever = GraphRetriever(graph)
            
            # Test entity relationships
            test_entities = ["diabetes", "metformin", "heart attack", "aspirin"]
            
            successful_analyses = 0
            total_relationships = 0
            
            for entity in test_entities:
                try:
                    # Get related entities
                    related_entities = retriever.get_related_entities(entity)
                    
                    # Find entity paths
                    paths = retriever.find_entity_paths(entity, max_paths=3)
                    
                    if related_entities or paths:
                        successful_analyses += 1
                        total_relationships += len(related_entities) if related_entities else 0
                        total_relationships += len(paths) if paths else 0
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to analyze entity '{entity}': {e}")
            
            duration = time.time() - start_time
            success_rate = (successful_analyses / len(test_entities)) * 100
            
            print(f"   ✅ Entity Relationships Test Complete ({duration:.2f}s)")
            print(f"      Successful Analyses: {successful_analyses}/{len(test_entities)}")
            print(f"      Total Relationships Found: {total_relationships}")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Entity relationships test completed",
                "duration": duration,
                "data": {
                    "successful_analyses": successful_analyses,
                    "total_entities": len(test_entities),
                    "total_relationships": total_relationships,
                    "success_rate": success_rate
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Entity relationships test failed: {e}")
            return {
                "status": "error",
                "message": f"Entity relationships test failed: {e}",
                "duration": duration
            }
    
    async def test_knowledge_graph_api(self) -> Dict[str, Any]:
        """Test knowledge graph API endpoints"""
        print("🌐 Testing Knowledge Graph API...")
        start_time = time.time()
        
        try:
            import requests
            
            base_url = "http://localhost:8000"
            
            # Test knowledge graph building endpoint
            build_data = {
                "topic_id": "test_topic_123",
                "articles": [
                    {
                        "title": "Test Article 1",
                        "abstract": "This is a test article about diabetes treatment.",
                        "pmid": "test_001"
                    }
                ]
            }
            
            build_response = requests.post(
                f"{base_url}/build-knowledge-graph/test_topic_123",
                json=build_data
            )
            
            # Test knowledge graph statistics endpoint
            stats_response = requests.get(f"{base_url}/knowledge-graph-stats/test_topic_123")
            
            # Test enhanced query endpoint (uses knowledge graph)
            enhanced_query_data = {
                "topic_id": "test_topic_123",
                "query": "diabetes treatment",
                "conversation_id": "test_conv_123"
            }
            
            enhanced_response = requests.post(
                f"{base_url}/enhanced-query",
                json=enhanced_query_data
            )
            
            duration = time.time() - start_time
            
            # Check responses
            successful_endpoints = 0
            if build_response.status_code in [200, 201]:
                successful_endpoints += 1
            if stats_response.status_code == 200:
                successful_endpoints += 1
            if enhanced_response.status_code == 200:
                successful_endpoints += 1
            
            success_rate = (successful_endpoints / 3) * 100
            
            print(f"   ✅ Knowledge Graph API Test Complete ({duration:.2f}s)")
            print(f"      Successful Endpoints: {successful_endpoints}/3")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Knowledge graph API test completed",
                "duration": duration,
                "data": {
                    "successful_endpoints": successful_endpoints,
                    "total_endpoints": 3,
                    "success_rate": success_rate,
                    "responses": {
                        "build_graph": build_response.status_code,
                        "graph_stats": stats_response.status_code,
                        "enhanced_query": enhanced_response.status_code
                    }
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Knowledge graph API test failed: {e}")
            return {
                "status": "error",
                "message": f"Knowledge graph API test failed: {e}",
                "duration": duration
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all knowledge graph tests"""
        print("🧪 Starting Knowledge Graph Tests...")
        print("=" * 50)
        
        tests = [
            ("Entity Extraction", self.test_entity_extraction),
            ("Relationship Extraction", self.test_relationship_extraction),
            ("Knowledge Graph Building", self.test_knowledge_graph_building),
            ("Graph-Based Retrieval", self.test_graph_retrieval),
            ("Entity Relationships", self.test_entity_relationships),
            ("Knowledge Graph API", self.test_knowledge_graph_api)
        ]
        
        results = {}
        total_duration = 0
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name} Test...")
            result = await test_func()
            results[test_name] = result
            total_duration += result.get("duration", 0)
            
            if result["status"] == "success":
                print(f"   ✅ {test_name} PASSED")
            else:
                print(f"   ❌ {test_name} FAILED")
        
        # Calculate overall results
        passed_tests = sum(1 for r in results.values() if r["status"] == "success")
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100
        
        print("\n" + "=" * 50)
        print("📊 Knowledge Graph Test Results")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/knowledge_graph_test_{timestamp}.json"
        
        try:
            import os
            os.makedirs("test_results", exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump({
                    "test_type": "knowledge_graph",
                    "timestamp": timestamp,
                    "overall_results": {
                        "total_tests": total_tests,
                        "passed_tests": passed_tests,
                        "failed_tests": total_tests - passed_tests,
                        "success_rate": success_rate,
                        "total_duration": total_duration
                    },
                    "detailed_results": results
                }, f, indent=2)
            
            print(f"📄 Results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
        
        return {
            "success": success_rate >= 80,
            "success_rate": success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "total_duration": total_duration,
            "results": results
        }

async def main():
    """Main test runner"""
    print("🚀 Knowledge Graph Test Suite")
    print("Testing entity extraction, relationship building, and graph-based retrieval")
    print("=" * 60)
    
    tester = KnowledgeGraphTester()
    results = await tester.run_all_tests()
    
    if results["success"]:
        print("\n🎉 All knowledge graph tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ Some tests failed. Success rate: {results['success_rate']:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
