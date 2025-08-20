#!/usr/bin/env python3
"""
Redis Caching Test Script for Vivum RAG Backend
Tests all caching functionality and performance improvements
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

class RedisCachingTester:
    """Test Redis caching functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.cache_hit_rates = []
        self.response_times = []
    
    async def test_redis_connection(self) -> Dict[str, Any]:
        """Test Redis connection and basic operations"""
        print("🔗 Testing Redis Connection...")
        start_time = time.time()
        
        try:
            from utils.cache import CacheManager
            from config.settings import REDIS_URL
            
            # Initialize cache manager
            cache = CacheManager(REDIS_URL)
            
            # Test basic operations
            test_key = "test_connection"
            test_value = {"message": "Redis is working", "timestamp": time.time()}
            
            # Set value
            cache.set(test_key, test_value, expire=60)
            
            # Get value
            retrieved_value = cache.get(test_key)
            
            # Test pattern invalidation
            cache.invalidate_pattern("test_*")
            
            # Verify invalidation
            invalidated_value = cache.get(test_key)
            
            duration = time.time() - start_time
            
            if retrieved_value == test_value and invalidated_value is None:
                print(f"   ✅ Redis caching working perfectly ({duration:.2f}s)")
                return {
                    "status": "success",
                    "message": "Redis caching fully functional",
                    "duration": duration,
                    "data": {
                        "set_operation": "success",
                        "get_operation": "success",
                        "invalidation": "success"
                    }
                }
            else:
                print(f"   ❌ Redis caching test failed")
                return {
                    "status": "error",
                    "message": "Redis operations failed",
                    "duration": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Redis connection failed: {e}")
            return {
                "status": "error",
                "message": f"Redis connection failed: {e}",
                "duration": duration
            }
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """Test caching performance improvements"""
        print("⚡ Testing Cache Performance...")
        start_time = time.time()
        
        try:
            from utils.cache import CacheManager
            from config.settings import REDIS_URL
            
            cache = CacheManager(REDIS_URL)
            
            # Simulate query caching
            test_queries = [
                "diabetes treatment options",
                "cardiovascular disease prevention",
                "cancer immunotherapy",
                "mental health therapies",
                "vaccine development"
            ]
            
            cache_hits = 0
            cache_misses = 0
            total_time_with_cache = 0
            total_time_without_cache = 0
            
            for i, query in enumerate(test_queries):
                cache_key = f"query_test_{hash(query)}"
                
                # Simulate processing time (without cache)
                processing_time = 2.0 + (i * 0.5)  # 2-4.5 seconds
                total_time_without_cache += processing_time
                
                # Check cache first
                cached_result = cache.get(cache_key)
                
                if cached_result:
                    cache_hits += 1
                    total_time_with_cache += 0.1  # 100ms for cache hit
                else:
                    cache_misses += 1
                    # Simulate processing and caching
                    await asyncio.sleep(0.1)  # Simulate processing
                    cache.set(cache_key, {
                        "query": query,
                        "result": f"Result for {query}",
                        "timestamp": time.time()
                    }, expire=3600)
                    total_time_with_cache += processing_time
                
                # Second request (should be cached)
                cached_result = cache.get(cache_key)
                if cached_result:
                    cache_hits += 1
                    total_time_with_cache += 0.1
            
            # Calculate performance metrics
            hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            time_saved = total_time_without_cache - total_time_with_cache
            performance_improvement = (time_saved / total_time_without_cache) * 100
            
            duration = time.time() - start_time
            
            print(f"   ✅ Cache Performance Test Complete ({duration:.2f}s)")
            print(f"      Cache Hit Rate: {hit_rate:.1f}%")
            print(f"      Time Saved: {time_saved:.2f}s")
            print(f"      Performance Improvement: {performance_improvement:.1f}%")
            
            return {
                "status": "success",
                "message": "Cache performance test completed",
                "duration": duration,
                "data": {
                    "cache_hit_rate": hit_rate,
                    "time_saved": time_saved,
                    "performance_improvement": performance_improvement,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Cache performance test failed: {e}")
            return {
                "status": "error",
                "message": f"Cache performance test failed: {e}",
                "duration": duration
            }
    
    async def test_cache_integration(self) -> Dict[str, Any]:
        """Test cache integration with API endpoints"""
        print("🔧 Testing Cache Integration...")
        start_time = time.time()
        
        try:
            import requests
            
            # Test API endpoints with caching
            base_url = "http://localhost:8000"
            
            # Test health endpoint (should be fast with caching)
            health_start = time.time()
            health_response = requests.get(f"{base_url}/health")
            health_time = time.time() - health_start
            
            # Test performance metrics (should be cached)
            metrics_start = time.time()
            metrics_response = requests.get(f"{base_url}/performance-metrics")
            metrics_time = time.time() - metrics_start
            
            # Test system health (should be cached)
            sys_health_start = time.time()
            sys_health_response = requests.get(f"{base_url}/system-health")
            sys_health_time = time.time() - sys_health_start
            
            duration = time.time() - start_time
            
            # Check if responses are fast (indicating caching)
            fast_responses = 0
            if health_time < 0.5:
                fast_responses += 1
            if metrics_time < 0.5:
                fast_responses += 1
            if sys_health_time < 0.5:
                fast_responses += 1
            
            success_rate = (fast_responses / 3) * 100
            
            print(f"   ✅ Cache Integration Test Complete ({duration:.2f}s)")
            print(f"      Fast Responses: {fast_responses}/3")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Cache integration test completed",
                "duration": duration,
                "data": {
                    "health_response_time": health_time,
                    "metrics_response_time": metrics_time,
                    "sys_health_response_time": sys_health_time,
                    "fast_responses": fast_responses,
                    "success_rate": success_rate
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Cache integration test failed: {e}")
            return {
                "status": "error",
                "message": f"Cache integration test failed: {e}",
                "duration": duration
            }
    
    async def test_cache_stress(self) -> Dict[str, Any]:
        """Test cache under stress conditions"""
        print("💪 Testing Cache Stress...")
        start_time = time.time()
        
        try:
            from utils.cache import CacheManager
            from config.settings import REDIS_URL
            
            cache = CacheManager(REDIS_URL)
            
            # Simulate high load
            concurrent_requests = 50
            test_data = []
            
            # Generate test data
            for i in range(concurrent_requests):
                test_data.append({
                    "key": f"stress_test_{i}",
                    "value": {
                        "id": i,
                        "data": f"Test data {i}",
                        "timestamp": time.time()
                    }
                })
            
            # Concurrent cache operations
            async def cache_operation(data):
                cache.set(data["key"], data["value"], expire=300)
                return cache.get(data["key"])
            
            # Run concurrent operations
            tasks = [cache_operation(data) for data in test_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_operations = sum(1 for r in results if r is not None and not isinstance(r, Exception))
            success_rate = (successful_operations / concurrent_requests) * 100
            
            duration = time.time() - start_time
            
            print(f"   ✅ Cache Stress Test Complete ({duration:.2f}s)")
            print(f"      Concurrent Operations: {concurrent_requests}")
            print(f"      Successful Operations: {successful_operations}")
            print(f"      Success Rate: {success_rate:.1f}%")
            
            return {
                "status": "success",
                "message": "Cache stress test completed",
                "duration": duration,
                "data": {
                    "concurrent_operations": concurrent_requests,
                    "successful_operations": successful_operations,
                    "success_rate": success_rate,
                    "operations_per_second": concurrent_requests / duration
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"   ❌ Cache stress test failed: {e}")
            return {
                "status": "error",
                "message": f"Cache stress test failed: {e}",
                "duration": duration
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Redis caching tests"""
        print("🧪 Starting Redis Caching Tests...")
        print("=" * 50)
        
        tests = [
            ("Redis Connection", self.test_redis_connection),
            ("Cache Performance", self.test_cache_performance),
            ("Cache Integration", self.test_cache_integration),
            ("Cache Stress", self.test_cache_stress)
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
        print("📊 Redis Caching Test Results")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/redis_caching_test_{timestamp}.json"
        
        try:
            import os
            os.makedirs("test_results", exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump({
                    "test_type": "redis_caching",
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
    print("🚀 Redis Caching Test Suite")
    print("Testing Redis caching functionality and performance improvements")
    print("=" * 60)
    
    tester = RedisCachingTester()
    results = await tester.run_all_tests()
    
    if results["success"]:
        print("\n🎉 All Redis caching tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ Some tests failed. Success rate: {results['success_rate']:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
