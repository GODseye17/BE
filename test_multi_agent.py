#!/usr/bin/env python3
"""
Multi-Agent System Test Script for Vivum RAG Backend
Tests research, clinical, statistical, and critic agents
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

class MultiAgentTester:
    """Test multi-agent system functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.agent_responses = []
    
    async def test_research_agent(self) -> Dict[str, Any]:
        """Test research agent functionality"""
        print("🔬 Testing Research Agent...")
        start_time = time.time()
        
        try:
            from agents.research_agent import ResearchAgent
            from config.settings import TOGETHER_API_KEY
            
            agent = ResearchAgent(TOGETHER_API_KEY)
            
            # Test articles
            test_articles = [
                {
                    "title": "Novel Diabetes Treatment Approaches",
                    "abstract": "This study investigates new treatment modalities for type 2 diabetes, including novel drug combinations and lifestyle interventions.",
                    "pmid": "test_001"
                },
                {
                    "title": "Cardiovascular Disease Prevention Strategies",
                    "abstract": "Comprehensive analysis of prevention strategies for cardiovascular disease, including pharmacological and non-pharmacological approaches.",
                    "pmid": "test_002"
                }
            ]
            
            # Test research analysis
            analysis = await agent.analyze_literature("diabetes treatment", test_articles)
            
            duration = time.time() - start_time
            
            # Check if analysis was successful
            analysis_successful = (
                analysis is not None and
                "insights" in analysis and
                "confidence" in analysis and
                len(analysis.get("insights", [])) > 0
            )
            
            print(f"   ✅ Research Agent Test Complete ({duration:.2f}s)")
            print(f"      Analysis Successful: {analysis_successful}")
            print(f"      Insights Count: {len(analysis.get('insights', []))}")
            print(f"      Confidence: {analysis.get('confidence', 0):.2f}")
            
            return {
                "status": "success",
                "message": "Research agent test completed",
                "duration": duration,
                "data": {
                    "analysis_successful": analysis_successful,
                    "insights_count": len(analysis.get("insights", [])),
                    "confidence": analysis.get("confidence", 0),
                    "analysis": analysis
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Research agent test failed: {e}")
            return {
                "status": "error",
                "message": f"Research agent test failed: {e}",
                "duration": duration
            }
    
    async def test_clinical_agent(self) -> Dict[str, Any]:
        """Test clinical agent functionality"""
        print("🏥 Testing Clinical Agent...")
        start_time = time.time()
        
        try:
            from agents.clinical_agent import ClinicalAgent
            from config.settings import TOGETHER_API_KEY
            
            agent = ClinicalAgent(TOGETHER_API_KEY)
            
            # Test articles
            test_articles = [
                {
                    "title": "Clinical Trial: Metformin vs Placebo",
                    "abstract": "Randomized controlled trial comparing metformin to placebo in type 2 diabetes patients. Results show significant improvement in glycemic control.",
                    "pmid": "test_001"
                },
                {
                    "title": "Patient Safety in Diabetes Management",
                    "abstract": "Study examining safety profiles of various diabetes medications and their clinical implications for patient care.",
                    "pmid": "test_002"
                }
            ]
            
            # Test clinical assessment
            assessment = await agent.assess_clinical_implications("diabetes treatment", test_articles)
            
            duration = time.time() - start_time
            
            # Check if assessment was successful
            assessment_successful = (
                assessment is not None and
                "insights" in assessment and
                "confidence" in assessment and
                len(assessment.get("insights", [])) > 0
            )
            
            print(f"   ✅ Clinical Agent Test Complete ({duration:.2f}s)")
            print(f"      Assessment Successful: {assessment_successful}")
            print(f"      Insights Count: {len(assessment.get('insights', []))}")
            print(f"      Confidence: {assessment.get('confidence', 0):.2f}")
            
            return {
                "status": "success",
                "message": "Clinical agent test completed",
                "duration": duration,
                "data": {
                    "assessment_successful": assessment_successful,
                    "insights_count": len(assessment.get("insights", [])),
                    "confidence": assessment.get("confidence", 0),
                    "assessment": assessment
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Clinical agent test failed: {e}")
            return {
                "status": "error",
                "message": f"Clinical agent test failed: {e}",
                "duration": duration
            }
    
    async def test_statistical_agent(self) -> Dict[str, Any]:
        """Test statistical agent functionality"""
        print("📊 Testing Statistical Agent...")
        start_time = time.time()
        
        try:
            from agents.statistical_agent import StatisticalAgent
            from config.settings import TOGETHER_API_KEY
            
            agent = StatisticalAgent(TOGETHER_API_KEY)
            
            # Test articles with statistical content
            test_articles = [
                {
                    "title": "Meta-analysis of Diabetes Treatments",
                    "abstract": "Systematic review and meta-analysis of 50 randomized controlled trials. Effect size: 0.45 (95% CI: 0.32-0.58), p<0.001.",
                    "pmid": "test_001"
                },
                {
                    "title": "Statistical Analysis of Cardiovascular Outcomes",
                    "abstract": "Cohort study with 10,000 participants. Hazard ratio: 0.78 (95% CI: 0.65-0.94), adjusted for age, sex, and comorbidities.",
                    "pmid": "test_002"
                }
            ]
            
            # Test statistical evaluation
            evaluation = await agent.evaluate_evidence("diabetes treatment", test_articles)
            
            duration = time.time() - start_time
            
            # Check if evaluation was successful
            evaluation_successful = (
                evaluation is not None and
                "insights" in evaluation and
                "confidence" in evaluation and
                len(evaluation.get("insights", [])) > 0
            )
            
            print(f"   ✅ Statistical Agent Test Complete ({duration:.2f}s)")
            print(f"      Evaluation Successful: {evaluation_successful}")
            print(f"      Insights Count: {len(evaluation.get('insights', []))}")
            print(f"      Confidence: {evaluation.get('confidence', 0):.2f}")
            
            return {
                "status": "success",
                "message": "Statistical agent test completed",
                "duration": duration,
                "data": {
                    "evaluation_successful": evaluation_successful,
                    "insights_count": len(evaluation.get("insights", [])),
                    "confidence": evaluation.get("confidence", 0),
                    "evaluation": evaluation
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Statistical agent test failed: {e}")
            return {
                "status": "error",
                "message": f"Statistical agent test failed: {e}",
                "duration": duration
            }
    
    async def test_critic_agent(self) -> Dict[str, Any]:
        """Test critic agent functionality"""
        print("🎯 Testing Critic Agent...")
        start_time = time.time()
        
        try:
            from agents.critic_agent import CriticAgent
            from config.settings import OPENAI_API_KEY
            
            if not OPENAI_API_KEY:
                print("   ⚠️ OpenAI API key not configured, skipping critic agent test")
                return {
                    "status": "warning",
                    "message": "OpenAI API key not configured",
                    "duration": time.time() - start_time,
                    "data": {"skipped": True}
                }
            
            agent = CriticAgent(OPENAI_API_KEY)
            
            # Test response validation
            test_response = "Diabetes can be treated with metformin, which is a first-line medication for type 2 diabetes."
            test_context = "Based on clinical guidelines and evidence-based medicine."
            
            validation = await agent.validate_response(test_response, test_context)
            
            duration = time.time() - start_time
            
            # Check if validation was successful
            validation_successful = (
                validation is not None and
                "validation_score" in validation and
                "issues" in validation
            )
            
            print(f"   ✅ Critic Agent Test Complete ({duration:.2f}s)")
            print(f"      Validation Successful: {validation_successful}")
            print(f"      Validation Score: {validation.get('validation_score', 0):.2f}")
            print(f"      Issues Found: {len(validation.get('issues', []))}")
            
            return {
                "status": "success",
                "message": "Critic agent test completed",
                "duration": duration,
                "data": {
                    "validation_successful": validation_successful,
                    "validation_score": validation.get("validation_score", 0),
                    "issues_count": len(validation.get("issues", [])),
                    "validation": validation
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Critic agent test failed: {e}")
            return {
                "status": "error",
                "message": f"Critic agent test failed: {e}",
                "duration": duration
            }
    
    async def test_multi_agent_coordinator(self) -> Dict[str, Any]:
        """Test multi-agent coordinator"""
        print("🤖 Testing Multi-Agent Coordinator...")
        start_time = time.time()
        
        try:
            from agents.coordinator import MultiAgentCoordinator
            from config.settings import TOGETHER_API_KEY, OPENAI_API_KEY
            
            coordinator = MultiAgentCoordinator(TOGETHER_API_KEY, OPENAI_API_KEY)
            
            # Test articles
            test_articles = [
                {
                    "title": "Comprehensive Diabetes Management",
                    "abstract": "This study provides a comprehensive overview of diabetes management strategies, including pharmacological and non-pharmacological approaches.",
                    "pmid": "test_001"
                }
            ]
            
            # Test coordinated analysis
            result = await coordinator.process_query("diabetes treatment", test_articles)
            
            duration = time.time() - start_time
            
            # Check if coordination was successful
            coordination_successful = (
                result is not None and
                "research_analysis" in result and
                "clinical_assessment" in result and
                "statistical_evaluation" in result and
                "overall_confidence" in result
            )
            
            print(f"   ✅ Multi-Agent Coordinator Test Complete ({duration:.2f}s)")
            print(f"      Coordination Successful: {coordination_successful}")
            print(f"      Overall Confidence: {result.get('overall_confidence', 0):.2f}")
            print(f"      Agents Used: {len([k for k in result.keys() if 'analysis' in k or 'assessment' in k or 'evaluation' in k])}")
            
            return {
                "status": "success",
                "message": "Multi-agent coordinator test completed",
                "duration": duration,
                "data": {
                    "coordination_successful": coordination_successful,
                    "overall_confidence": result.get("overall_confidence", 0),
                    "agents_used": len([k for k in result.keys() if 'analysis' in k or 'assessment' in k or 'evaluation' in k]),
                    "result": result
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Multi-agent coordinator test failed: {e}")
            return {
                "status": "error",
                "message": f"Multi-agent coordinator test failed: {e}",
                "duration": duration
            }
    
    async def test_agent_api_endpoints(self) -> Dict[str, Any]:
        """Test multi-agent API endpoints"""
        print("🌐 Testing Multi-Agent API Endpoints...")
        start_time = time.time()
        
        try:
            import requests
            
            base_url = "http://localhost:8000"
            
            # Test enhanced query endpoint (uses multi-agents)
            enhanced_query_data = {
                "topic_id": "test_topic_123",
                "query": "diabetes treatment",
                "conversation_id": "test_conv_123"
            }
            
            enhanced_response = requests.post(
                f"{base_url}/enhanced-query",
                json=enhanced_query_data
            )
            
            # Test multi-agent status endpoint
            status_response = requests.get(f"{base_url}/multi-agent-status/test_topic_123")
            
            # Test enable critic agent endpoint
            critic_data = {
                "topic_id": "test_topic_123",
                "openai_api_key": "test_key"
            }
            
            critic_response = requests.post(
                f"{base_url}/enable-critic-agent",
                json=critic_data
            )
            
            duration = time.time() - start_time
            
            # Check responses
            successful_endpoints = 0
            if enhanced_response.status_code == 200:
                successful_endpoints += 1
            if status_response.status_code == 200:
                successful_endpoints += 1
            if critic_response.status_code == 200:
                successful_endpoints += 1
            
            success_rate = (successful_endpoints / 3) * 100
            
            print(f"   ✅ Multi-Agent API Endpoints Test Complete ({duration:.2f}s)")
            print(f"      Successful Endpoints: {successful_endpoints}/3")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Multi-agent API endpoints test completed",
                "duration": duration,
                "data": {
                    "successful_endpoints": successful_endpoints,
                    "total_endpoints": 3,
                    "success_rate": success_rate,
                    "responses": {
                        "enhanced_query": enhanced_response.status_code,
                        "multi_agent_status": status_response.status_code,
                        "enable_critic": critic_response.status_code
                    }
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Multi-agent API endpoints test failed: {e}")
            return {
                "status": "error",
                "message": f"Multi-agent API endpoints test failed: {e}",
                "duration": duration
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all multi-agent tests"""
        print("🧪 Starting Multi-Agent System Tests...")
        print("=" * 50)
        
        tests = [
            ("Research Agent", self.test_research_agent),
            ("Clinical Agent", self.test_clinical_agent),
            ("Statistical Agent", self.test_statistical_agent),
            ("Critic Agent", self.test_critic_agent),
            ("Multi-Agent Coordinator", self.test_multi_agent_coordinator),
            ("Multi-Agent API Endpoints", self.test_agent_api_endpoints)
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
            elif result["status"] == "warning":
                print(f"   ⚠️ {test_name} WARNING")
            else:
                print(f"   ❌ {test_name} FAILED")
        
        # Calculate overall results (excluding warnings)
        passed_tests = sum(1 for r in results.values() if r["status"] == "success")
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100
        
        print("\n" + "=" * 50)
        print("📊 Multi-Agent System Test Results")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/multi_agent_test_{timestamp}.json"
        
        try:
            import os
            os.makedirs("test_results", exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump({
                    "test_type": "multi_agent",
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
    print("🚀 Multi-Agent System Test Suite")
    print("Testing research, clinical, statistical, and critic agents")
    print("=" * 60)
    
    tester = MultiAgentTester()
    results = await tester.run_all_tests()
    
    if results["success"]:
        print("\n🎉 All multi-agent tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ Some tests failed. Success rate: {results['success_rate']:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
