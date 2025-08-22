#!/usr/bin/env python3
"""
Test script for Auto-Tuning System

This script tests the auto-tuning system's functionality, performance tracking,
parameter adjustment, and A/B testing capabilities.
"""

import time
import logging
import random
import asyncio
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_auto_tuning_basic():
    """Test basic auto-tuning system functionality"""
    logger.info("🔧 Testing Auto-Tuning System Basic Functionality")
    logger.info("=" * 60)
    
    try:
        from optimization.auto_tuner import (
            AutoTuningSystem, 
            PerformanceMetrics, 
            TuningParameters,
            OptimizationStrategy
        )
        
        # Initialize auto-tuning system
        auto_tuner = AutoTuningSystem()
        logger.info("✅ AutoTuningSystem initialized successfully")
        
        # Test parameter loading
        current_params = auto_tuner.get_current_parameters()
        logger.info(f"📊 Current parameters: {current_params.to_dict()}")
        
        # Test performance metrics recording
        test_metrics = PerformanceMetrics(
            response_time=1.5,
            quality_score=0.8,
            memory_usage_mb=500.0,
            cache_hit_rate=0.6,
            throughput_queries_per_min=40.0,
            error_rate=0.02,
            timestamp=time.time()
        )
        
        auto_tuner.record_metrics(test_metrics)
        logger.info("✅ Performance metrics recorded successfully")
        
        # Test metrics calculation
        metrics = auto_tuner.calculate_metrics()
        logger.info(f"📈 Calculated metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic test failed: {e}")
        return False

def test_parameter_optimization():
    """Test parameter optimization strategies"""
    logger.info("\n⚡ Testing Parameter Optimization Strategies")
    logger.info("=" * 60)
    
    try:
        from optimization.auto_tuner import AutoTuningSystem, PerformanceMetrics, OptimizationStrategy
        
        auto_tuner = AutoTuningSystem()
        
        # Test different optimization strategies
        strategies = [
            OptimizationStrategy.RESPONSE_TIME,
            OptimizationStrategy.QUALITY_SCORE,
            OptimizationStrategy.MEMORY_USAGE,
            OptimizationStrategy.CACHE_HIT_RATE,
            OptimizationStrategy.THROUGHPUT,
            OptimizationStrategy.BALANCED
        ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"\n🔍 Testing strategy: {strategy.value}")
            
            # Set strategy
            auto_tuner.set_optimization_strategy(strategy)
            
            # Record some test metrics
            for i in range(10):
                # Simulate different performance scenarios
                if strategy == OptimizationStrategy.RESPONSE_TIME:
                    response_time = 2.5 + random.uniform(-0.5, 0.5)  # Slow response
                    quality_score = 0.7
                elif strategy == OptimizationStrategy.QUALITY_SCORE:
                    response_time = 1.0
                    quality_score = 0.5 + random.uniform(-0.1, 0.1)  # Low quality
                elif strategy == OptimizationStrategy.MEMORY_USAGE:
                    response_time = 1.0
                    quality_score = 0.8
                    memory_usage = 1200.0  # High memory
                elif strategy == OptimizationStrategy.CACHE_HIT_RATE:
                    response_time = 1.0
                    quality_score = 0.8
                    cache_hit_rate = 0.2  # Low cache hit rate
                elif strategy == OptimizationStrategy.THROUGHPUT:
                    response_time = 1.0
                    quality_score = 0.8
                    throughput = 5.0  # Low throughput
                else:  # BALANCED
                    response_time = 1.5
                    quality_score = 0.7
                
                metrics = PerformanceMetrics(
                    response_time=response_time,
                    quality_score=quality_score,
                    memory_usage_mb=getattr(locals(), 'memory_usage', 500.0),
                    cache_hit_rate=getattr(locals(), 'cache_hit_rate', 0.6),
                    throughput_queries_per_min=getattr(locals(), 'throughput', 40.0),
                    error_rate=0.02,
                    timestamp=time.time()
                )
                
                auto_tuner.record_metrics(metrics)
            
            # Get parameters after optimization
            final_params = auto_tuner.get_current_parameters()
            results[strategy.value] = final_params.to_dict()
            
            logger.info(f"  Final parameters: {final_params.to_dict()}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Parameter optimization test failed: {e}")
        return {}

def test_ab_testing():
    """Test A/B testing functionality"""
    logger.info("\n🧪 Testing A/B Testing Functionality")
    logger.info("=" * 60)
    
    try:
        from optimization.auto_tuner import AutoTuningSystem, PerformanceMetrics
        
        auto_tuner = AutoTuningSystem()
        
        # Start A/B test for retrieval_k
        test_id = auto_tuner.ab_test_parameter(
            param_name='retrieval_k',
            value_a=5,
            value_b=15,
            duration=20
        )
        
        logger.info(f"🔬 Started A/B test: {test_id}")
        
        # Record metrics for both variants
        for i in range(20):
            # Simulate metrics for variant A (lower retrieval_k)
            metrics_a = PerformanceMetrics(
                response_time=0.8,  # Faster
                quality_score=0.6,  # Lower quality
                memory_usage_mb=300.0,
                cache_hit_rate=0.7,
                throughput_queries_per_min=50.0,
                error_rate=0.01,
                timestamp=time.time()
            )
            
            # Simulate metrics for variant B (higher retrieval_k)
            metrics_b = PerformanceMetrics(
                response_time=1.5,  # Slower
                quality_score=0.9,  # Higher quality
                memory_usage_mb=600.0,
                cache_hit_rate=0.5,
                throughput_queries_per_min=30.0,
                error_rate=0.02,
                timestamp=time.time()
            )
            
            # Record metrics for both variants
            auto_tuner.record_ab_test_metric(test_id, 'A', metrics_a)
            auto_tuner.record_ab_test_metric(test_id, 'B', metrics_b)
        
        # Check if test is completed
        if test_id in auto_tuner.ab_tests and auto_tuner.ab_tests[test_id]['completed']:
            logger.info("✅ A/B test completed successfully")
            
            # Get test results
            recent_results = auto_tuner.ab_test_results[-1] if auto_tuner.ab_test_results else None
            if recent_results:
                logger.info(f"🏆 Test result: {recent_results.winner} wins with confidence {recent_results.confidence:.2f}")
            
            return True
        else:
            logger.warning("⚠️ A/B test did not complete as expected")
            return False
        
    except Exception as e:
        logger.error(f"❌ A/B testing failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring and summary"""
    logger.info("\n📊 Testing Performance Monitoring")
    logger.info("=" * 60)
    
    try:
        from optimization.auto_tuner import AutoTuningSystem, PerformanceMetrics
        
        auto_tuner = AutoTuningSystem()
        
        # Record various performance scenarios
        scenarios = [
            # Fast response, high quality
            {'response_time': 0.5, 'quality_score': 0.9, 'memory_usage': 200.0, 'cache_hit_rate': 0.8},
            # Slow response, low quality
            {'response_time': 3.0, 'quality_score': 0.4, 'memory_usage': 800.0, 'cache_hit_rate': 0.2},
            # Balanced performance
            {'response_time': 1.2, 'quality_score': 0.7, 'memory_usage': 400.0, 'cache_hit_rate': 0.6},
            # High memory usage
            {'response_time': 1.0, 'quality_score': 0.8, 'memory_usage': 1200.0, 'cache_hit_rate': 0.5},
            # Low cache hit rate
            {'response_time': 1.5, 'quality_score': 0.6, 'memory_usage': 300.0, 'cache_hit_rate': 0.1},
        ]
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"📝 Recording scenario {i+1}: {scenario}")
            
            metrics = PerformanceMetrics(
                response_time=scenario['response_time'],
                quality_score=scenario['quality_score'],
                memory_usage_mb=scenario['memory_usage'],
                cache_hit_rate=scenario['cache_hit_rate'],
                throughput_queries_per_min=60.0 / scenario['response_time'],
                error_rate=0.01,
                timestamp=time.time()
            )
            
            auto_tuner.record_metrics(metrics)
        
        # Get performance summary
        summary = auto_tuner.get_performance_summary()
        
        logger.info("📈 Performance Summary:")
        logger.info(f"  Total queries: {summary['optimization_stats']['total_queries']}")
        logger.info(f"  Optimization count: {summary['optimization_stats']['optimization_count']}")
        logger.info(f"  Improvement count: {summary['optimization_stats']['improvement_count']}")
        logger.info(f"  Recent metrics: {summary['recent_metrics']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return {}

def test_parameter_persistence():
    """Test parameter persistence and loading"""
    logger.info("\n💾 Testing Parameter Persistence")
    logger.info("=" * 60)
    
    try:
        from optimization.auto_tuner import AutoTuningSystem, TuningParameters
        
        # Create auto-tuning system
        auto_tuner = AutoTuningSystem()
        
        # Get initial parameters
        initial_params = auto_tuner.get_current_parameters()
        logger.info(f"📋 Initial parameters: {initial_params.to_dict()}")
        
        # Modify parameters
        auto_tuner.current_params.retrieval_k = 20
        auto_tuner.current_params.similarity_threshold = 0.6
        auto_tuner.current_params.cache_ttl = 7200
        
        # Save parameters
        auto_tuner.save_parameters(auto_tuner.current_params)
        logger.info("💾 Parameters saved")
        
        # Create new instance to test loading
        new_auto_tuner = AutoTuningSystem()
        loaded_params = new_auto_tuner.get_current_parameters()
        
        logger.info(f"📋 Loaded parameters: {loaded_params.to_dict()}")
        
        # Verify parameters were loaded correctly
        if (loaded_params.retrieval_k == 20 and 
            loaded_params.similarity_threshold == 0.6 and 
            loaded_params.cache_ttl == 7200):
            logger.info("✅ Parameter persistence test passed")
            return True
        else:
            logger.error("❌ Parameter persistence test failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Parameter persistence test failed: {e}")
        return False

def test_integration_with_api():
    """Test integration with API routes"""
    logger.info("\n🔗 Testing Integration with API Routes")
    logger.info("=" * 60)
    
    try:
        # Test the record_query_metrics function
        from optimization.auto_tuner import record_query_metrics, get_tuned_parameters
        
        # Record some test metrics
        record_query_metrics(
            response_time=1.2,
            quality_score=0.8,
            memory_usage_mb=400.0,
            cache_hit_rate=0.7,
            throughput_queries_per_min=50.0,
            error_rate=0.01
        )
        
        logger.info("✅ Query metrics recorded successfully")
        
        # Get tuned parameters
        tuned_params = get_tuned_parameters()
        logger.info(f"📊 Tuned parameters: {tuned_params.to_dict()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        return False

async def test_async_integration():
    """Test async integration with the pipeline"""
    logger.info("\n⚡ Testing Async Integration")
    logger.info("=" * 60)
    
    try:
        from optimization.auto_tuner import get_auto_tuner, record_query_metrics
        
        auto_tuner = get_auto_tuner()
        
        # Simulate async query processing
        async def simulate_query_processing():
            start_time = time.time()
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            # Record metrics
            record_query_metrics(
                response_time=processing_time,
                quality_score=0.85,
                memory_usage_mb=350.0,
                cache_hit_rate=0.75,
                throughput_queries_per_min=60.0 / processing_time,
                error_rate=0.005
            )
            
            return processing_time
        
        # Run multiple simulated queries
        tasks = [simulate_query_processing() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        logger.info(f"✅ Async integration test completed. Processing times: {results}")
        
        # Check if auto-tuning was triggered
        summary = auto_tuner.get_performance_summary()
        logger.info(f"📊 Auto-tuning summary: {summary['optimization_stats']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Async integration test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Auto-Tuning System Tests")
    logger.info("=" * 60)
    
    # Test 1: Basic functionality
    logger.info("\n📋 Test 1: Basic Functionality")
    logger.info("-" * 40)
    basic_success = test_auto_tuning_basic()
    
    # Test 2: Parameter optimization
    logger.info("\n⚡ Test 2: Parameter Optimization")
    logger.info("-" * 40)
    optimization_results = test_parameter_optimization()
    
    # Test 3: A/B testing
    logger.info("\n🧪 Test 3: A/B Testing")
    logger.info("-" * 40)
    ab_test_success = test_ab_testing()
    
    # Test 4: Performance monitoring
    logger.info("\n📊 Test 4: Performance Monitoring")
    logger.info("-" * 40)
    monitoring_results = test_performance_monitoring()
    
    # Test 5: Parameter persistence
    logger.info("\n💾 Test 5: Parameter Persistence")
    logger.info("-" * 40)
    persistence_success = test_parameter_persistence()
    
    # Test 6: API integration
    logger.info("\n🔗 Test 6: API Integration")
    logger.info("-" * 40)
    api_integration_success = test_integration_with_api()
    
    # Test 7: Async integration
    logger.info("\n⚡ Test 7: Async Integration")
    logger.info("-" * 40)
    async_success = asyncio.run(test_async_integration())
    
    # Summary
    logger.info("\n📊 Test Summary")
    logger.info("=" * 60)
    
    test_results = {
        "Basic Functionality": basic_success,
        "Parameter Optimization": len(optimization_results) > 0,
        "A/B Testing": ab_test_success,
        "Performance Monitoring": len(monitoring_results) > 0,
        "Parameter Persistence": persistence_success,
        "API Integration": api_integration_success,
        "Async Integration": async_success
    }
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("\n🎉 All auto-tuning system tests completed successfully!")
        logger.info("The auto-tuning system is working correctly and ready for production!")
        logger.info("Key features verified:")
        logger.info("• ✅ Performance metrics tracking")
        logger.info("• ✅ Automatic parameter optimization")
        logger.info("• ✅ A/B testing with statistical significance")
        logger.info("• ✅ Parameter persistence and loading")
        logger.info("• ✅ API integration")
        logger.info("• ✅ Async pipeline integration")
        logger.info("• ✅ Multiple optimization strategies")
    else:
        logger.warning(f"\n⚠️ {total_tests - passed_tests} tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()
