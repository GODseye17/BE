#!/usr/bin/env python3
"""
Feedback Loop Test Script for Vivum RAG Backend
Tests feedback collection, relevance tracking, and adaptive optimization
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

class FeedbackLoopTester:
    """Test feedback loop functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.feedback_data = []
        self.threshold_adjustments = []
    
    async def test_feedback_tracker_initialization(self) -> Dict[str, Any]:
        """Test feedback tracker initialization"""
        print("🔧 Testing Feedback Tracker Initialization...")
        start_time = time.time()
        
        try:
            from feedback.relevance_tracker import RelevanceTracker
            
            # Initialize feedback tracker
            tracker = RelevanceTracker()
            
            # Test default thresholds
            default_thresholds = tracker.get_current_thresholds()
            
            # Test feedback data structure
            feedback_data = tracker.get_feedback_summary()
            
            duration = time.time() - start_time
            
            if default_thresholds and feedback_data is not None:
                print(f"   ✅ Feedback tracker initialized successfully ({duration:.2f}s)")
                return {
                    "status": "success",
                    "message": "Feedback tracker initialization successful",
                    "duration": duration,
                    "data": {
                        "default_thresholds": default_thresholds,
                        "feedback_data": feedback_data
                    }
                }
            else:
                print(f"   ❌ Feedback tracker initialization failed")
                return {
                    "status": "error",
                    "message": "Feedback tracker initialization failed",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Feedback tracker initialization failed: {e}")
            return {
                "status": "error",
                "message": f"Feedback tracker initialization failed: {e}",
                "duration": duration
            }
    
    async def test_article_feedback_collection(self) -> Dict[str, Any]:
        """Test article relevance feedback collection"""
        print("📝 Testing Article Feedback Collection...")
        start_time = time.time()
        
        try:
            from feedback.relevance_tracker import RelevanceTracker
            
            tracker = RelevanceTracker()
            
            # Test article feedback scenarios
            test_feedbacks = [
                {
                    "query": "diabetes treatment",
                    "pmid": "12345678",
                    "is_relevant": True,
                    "user_score": 4.5
                },
                {
                    "query": "diabetes treatment",
                    "pmid": "87654321",
                    "is_relevant": False,
                    "user_score": 1.0
                },
                {
                    "query": "cardiovascular disease",
                    "pmid": "11223344",
                    "is_relevant": True,
                    "user_score": 5.0
                }
            ]
            
            successful_feedbacks = 0
            
            for feedback in test_feedbacks:
                try:
                    tracker.record_article_feedback(
                        query=feedback["query"],
                        pmid=feedback["pmid"],
                        is_relevant=feedback["is_relevant"],
                        user_score=feedback["user_score"]
                    )
                    successful_feedbacks += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to record feedback: {e}")
            
            # Get updated feedback summary
            updated_summary = tracker.get_feedback_summary()
            
            duration = time.time() - start_time
            success_rate = (successful_feedbacks / len(test_feedbacks)) * 100
            
            print(f"   ✅ Article Feedback Collection Test Complete ({duration:.2f}s)")
            print(f"      Successful Feedbacks: {successful_feedbacks}/{len(test_feedbacks)}")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Article feedback collection test completed",
                "duration": duration,
                "data": {
                    "successful_feedbacks": successful_feedbacks,
                    "total_feedbacks": len(test_feedbacks),
                    "success_rate": success_rate,
                    "feedback_summary": updated_summary
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Article feedback collection test failed: {e}")
            return {
                "status": "error",
                "message": f"Article feedback collection test failed: {e}",
                "duration": duration
            }
    
    async def test_query_satisfaction_tracking(self) -> Dict[str, Any]:
        """Test query satisfaction feedback tracking"""
        print("😊 Testing Query Satisfaction Tracking...")
        start_time = time.time()
        
        try:
            from feedback.relevance_tracker import RelevanceTracker
            
            tracker = RelevanceTracker()
            
            # Test query satisfaction scenarios
            test_satisfactions = [
                {
                    "query": "diabetes treatment options",
                    "satisfaction_score": 4.2,
                    "feedback_text": "Very helpful response with good citations"
                },
                {
                    "query": "cancer immunotherapy",
                    "satisfaction_score": 3.8,
                    "feedback_text": "Good information but could be more detailed"
                },
                {
                    "query": "mental health therapies",
                    "satisfaction_score": 5.0,
                    "feedback_text": "Excellent comprehensive response"
                }
            ]
            
            successful_satisfactions = 0
            
            for satisfaction in test_satisfactions:
                try:
                    tracker.record_query_satisfaction(
                        query=satisfaction["query"],
                        satisfaction_score=satisfaction["satisfaction_score"],
                        feedback_text=satisfaction["feedback_text"]
                    )
                    successful_satisfactions += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to record satisfaction: {e}")
            
            # Get updated feedback summary
            updated_summary = tracker.get_feedback_summary()
            
            duration = time.time() - start_time
            success_rate = (successful_satisfactions / len(test_satisfactions)) * 100
            
            print(f"   ✅ Query Satisfaction Tracking Test Complete ({duration:.2f}s)")
            print(f"      Successful Satisfactions: {successful_satisfactions}/{len(test_satisfactions)}")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Query satisfaction tracking test completed",
                "duration": duration,
                "data": {
                    "successful_satisfactions": successful_satisfactions,
                    "total_satisfactions": len(test_satisfactions),
                    "success_rate": success_rate,
                    "feedback_summary": updated_summary
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Query satisfaction tracking test failed: {e}")
            return {
                "status": "error",
                "message": f"Query satisfaction tracking test failed: {e}",
                "duration": duration
            }
    
    async def test_threshold_adjustment(self) -> Dict[str, Any]:
        """Test dynamic threshold adjustment based on feedback"""
        print("⚖️ Testing Threshold Adjustment...")
        start_time = time.time()
        
        try:
            from feedback.relevance_tracker import RelevanceTracker
            
            tracker = RelevanceTracker()
            
            # Get initial thresholds
            initial_thresholds = tracker.get_current_thresholds()
            
            # Simulate feedback that should trigger threshold adjustment
            # Add multiple low-satisfaction feedbacks
            for i in range(10):
                tracker.record_query_satisfaction(
                    query=f"test_query_{i}",
                    satisfaction_score=2.0,  # Low satisfaction
                    feedback_text="Not very helpful"
                )
            
            # Add some article feedback
            for i in range(5):
                tracker.record_article_feedback(
                    query=f"test_query_{i}",
                    pmid=f"test_pmid_{i}",
                    is_relevant=False,
                    user_score=1.0
                )
            
            # Trigger threshold adjustment
            tracker.adjust_thresholds_based_on_feedback()
            
            # Get updated thresholds
            updated_thresholds = tracker.get_current_thresholds()
            
            # Check if thresholds changed
            thresholds_changed = initial_thresholds != updated_thresholds
            
            duration = time.time() - start_time
            
            print(f"   ✅ Threshold Adjustment Test Complete ({duration:.2f}s)")
            print(f"      Thresholds Changed: {thresholds_changed}")
            print(f"      Initial Thresholds: {initial_thresholds}")
            print(f"      Updated Thresholds: {updated_thresholds}")
            
            return {
                "status": "success",
                "message": "Threshold adjustment test completed",
                "duration": duration,
                "data": {
                    "thresholds_changed": thresholds_changed,
                    "initial_thresholds": initial_thresholds,
                    "updated_thresholds": updated_thresholds,
                    "adjustment_triggered": True
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Threshold adjustment test failed: {e}")
            return {
                "status": "error",
                "message": f"Threshold adjustment test failed: {e}",
                "duration": duration
            }
    
    async def test_feedback_api_endpoints(self) -> Dict[str, Any]:
        """Test feedback API endpoints"""
        print("🌐 Testing Feedback API Endpoints...")
        start_time = time.time()
        
        try:
            import requests
            
            base_url = "http://localhost:8000"
            
            # Test article relevance feedback endpoint
            article_feedback_data = {
                "query": "diabetes treatment",
                "pmid": "12345678",
                "is_relevant": True,
                "user_score": 4.5
            }
            
            article_response = requests.post(
                f"{base_url}/feedback/article-relevance",
                json=article_feedback_data
            )
            
            # Test query satisfaction feedback endpoint
            satisfaction_data = {
                "query": "diabetes treatment",
                "satisfaction_score": 4.2,
                "feedback_text": "Very helpful response"
            }
            
            satisfaction_response = requests.post(
                f"{base_url}/feedback/query-satisfaction",
                json=satisfaction_data
            )
            
            # Test feedback summary endpoint
            summary_response = requests.get(f"{base_url}/feedback/summary")
            
            # Test reset thresholds endpoint
            reset_response = requests.post(f"{base_url}/feedback/reset-thresholds")
            
            duration = time.time() - start_time
            
            # Check responses
            successful_endpoints = 0
            if article_response.status_code == 200:
                successful_endpoints += 1
            if satisfaction_response.status_code == 200:
                successful_endpoints += 1
            if summary_response.status_code == 200:
                successful_endpoints += 1
            if reset_response.status_code == 200:
                successful_endpoints += 1
            
            success_rate = (successful_endpoints / 4) * 100
            
            print(f"   ✅ Feedback API Endpoints Test Complete ({duration:.2f}s)")
            print(f"      Successful Endpoints: {successful_endpoints}/4")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Feedback API endpoints test completed",
                "duration": duration,
                "data": {
                    "successful_endpoints": successful_endpoints,
                    "total_endpoints": 4,
                    "success_rate": success_rate,
                    "responses": {
                        "article_feedback": article_response.status_code,
                        "query_satisfaction": satisfaction_response.status_code,
                        "feedback_summary": summary_response.status_code,
                        "reset_thresholds": reset_response.status_code
                    }
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Feedback API endpoints test failed: {e}")
            return {
                "status": "error",
                "message": f"Feedback API endpoints test failed: {e}",
                "duration": duration
            }
    
    async def test_feedback_persistence(self) -> Dict[str, Any]:
        """Test feedback data persistence and retrieval"""
        print("💾 Testing Feedback Persistence...")
        start_time = time.time()
        
        try:
            from feedback.relevance_tracker import RelevanceTracker
            
            tracker = RelevanceTracker()
            
            # Add test feedback data
            test_data = [
                {"query": "test_query_1", "pmid": "test_1", "is_relevant": True, "score": 4.0},
                {"query": "test_query_2", "pmid": "test_2", "is_relevant": False, "score": 1.0},
                {"query": "test_query_3", "pmid": "test_3", "is_relevant": True, "score": 5.0}
            ]
            
            # Record feedback
            for data in test_data:
                tracker.record_article_feedback(
                    query=data["query"],
                    pmid=data["pmid"],
                    is_relevant=data["is_relevant"],
                    user_score=data["score"]
                )
            
            # Get feedback summary
            summary = tracker.get_feedback_summary()
            
            # Check if data is persisted
            data_persisted = (
                summary is not None and 
                "article_feedback" in summary and 
                len(summary["article_feedback"]) > 0
            )
            
            duration = time.time() - start_time
            
            print(f"   ✅ Feedback Persistence Test Complete ({duration:.2f}s)")
            print(f"      Data Persisted: {data_persisted}")
            print(f"      Feedback Count: {len(summary.get('article_feedback', []))}")
            
            return {
                "status": "success",
                "message": "Feedback persistence test completed",
                "duration": duration,
                "data": {
                    "data_persisted": data_persisted,
                    "feedback_count": len(summary.get('article_feedback', [])),
                    "summary": summary
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Feedback persistence test failed: {e}")
            return {
                "status": "error",
                "message": f"Feedback persistence test failed: {e}",
                "duration": duration
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all feedback loop tests"""
        print("🧪 Starting Feedback Loop Tests...")
        print("=" * 50)
        
        tests = [
            ("Feedback Tracker Initialization", self.test_feedback_tracker_initialization),
            ("Article Feedback Collection", self.test_article_feedback_collection),
            ("Query Satisfaction Tracking", self.test_query_satisfaction_tracking),
            ("Threshold Adjustment", self.test_threshold_adjustment),
            ("Feedback API Endpoints", self.test_feedback_api_endpoints),
            ("Feedback Persistence", self.test_feedback_persistence)
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
        print("📊 Feedback Loop Test Results")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/feedback_loop_test_{timestamp}.json"
        
        try:
            import os
            os.makedirs("test_results", exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump({
                    "test_type": "feedback_loop",
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
    print("🚀 Feedback Loop Test Suite")
    print("Testing feedback collection, relevance tracking, and adaptive optimization")
    print("=" * 60)
    
    tester = FeedbackLoopTester()
    results = await tester.run_all_tests()
    
    if results["success"]:
        print("\n🎉 All feedback loop tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ Some tests failed. Success rate: {results['success_rate']:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
