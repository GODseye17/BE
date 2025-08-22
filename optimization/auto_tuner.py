"""
Auto-Tuning System for Performance Optimization

This module implements an intelligent auto-tuning system that automatically
adjusts system parameters based on performance metrics to achieve optimal
performance without manual intervention.
"""

import json
import logging
import time
import statistics
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import threading
import asyncio

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    RESPONSE_TIME = "response_time"
    QUALITY_SCORE = "quality_score"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"

@dataclass
class PerformanceMetrics:
    """Performance metrics for auto-tuning"""
    response_time: float
    quality_score: float
    memory_usage_mb: float
    cache_hit_rate: float
    throughput_queries_per_min: float
    error_rate: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TuningParameters:
    """System parameters that can be auto-tuned"""
    retrieval_k: int = 10
    rerank_k: int = 5
    chunk_size: int = 500
    similarity_threshold: float = 0.5
    cache_ttl: int = 3600
    batch_size: int = 50
    max_concurrent_queries: int = 20
    lego_extraction_method: str = "PPR"
    query_optimization_enabled: bool = True
    circuit_breaker_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TuningParameters':
        return cls(**data)

@dataclass
class ABTestResult:
    """A/B test result"""
    param_name: str
    value_a: Any
    value_b: Any
    metric_a: float
    metric_b: float
    confidence: float
    winner: str
    sample_size: int
    duration: float

class AutoTuningSystem:
    """
    Intelligent auto-tuning system for performance optimization
    
    Automatically adjusts system parameters based on performance metrics
    to achieve optimal performance without manual intervention.
    """
    
    # Tunable parameters with their ranges
    TUNABLE_PARAMS = {
        'retrieval_k': (5, 30),
        'rerank_k': (3, 15),
        'chunk_size': (200, 1000),
        'similarity_threshold': (0.3, 0.7),
        'cache_ttl': (300, 7200),
        'batch_size': (10, 100),
        'max_concurrent_queries': (10, 50),
    }
    
    # Performance thresholds for auto-tuning
    PERFORMANCE_THRESHOLDS = {
        'response_time_fast': 0.5,      # seconds
        'response_time_slow': 2.0,      # seconds
        'quality_score_low': 0.6,       # 0-1 scale
        'quality_score_high': 0.8,      # 0-1 scale
        'cache_hit_rate_low': 0.3,      # 0-1 scale
        'cache_hit_rate_high': 0.7,     # 0-1 scale
        'memory_usage_high': 1000,      # MB
        'error_rate_high': 0.05,        # 0-1 scale
    }
    
    def __init__(self, config_path: str = "config/auto_tuning.json"):
        """
        Initialize the auto-tuning system
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.recent_metrics = deque(maxlen=100)  # Last 100 queries
        
        # Current parameters
        self.current_params = self.load_or_default_params()
        self.previous_params = self.current_params.to_dict()
        
        # Optimization settings
        self.optimization_interval = 100  # queries
        self.query_count = 0
        self.last_optimization = time.time()
        
        # A/B testing
        self.ab_tests = {}
        self.ab_test_results = []
        
        # Performance targets
        self.optimization_strategy = OptimizationStrategy.BALANCED
        self.performance_targets = {
            'target_response_time': 1.0,
            'target_quality_score': 0.8,
            'target_cache_hit_rate': 0.6,
            'max_memory_usage': 800,  # MB
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.optimization_count = 0
        self.improvement_count = 0
        
        logger.info(f"AutoTuningSystem initialized with {len(self.TUNABLE_PARAMS)} tunable parameters")
    
    def load_or_default_params(self) -> TuningParameters:
        """Load parameters from config file or use defaults"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    params = TuningParameters.from_dict(data)
                    logger.info(f"Loaded parameters from {self.config_path}")
                    return params
        except Exception as e:
            logger.warning(f"Failed to load parameters: {e}, using defaults")
        
        # Default parameters
        default_params = TuningParameters()
        self.save_parameters(default_params)
        return default_params
    
    def save_parameters(self, params: TuningParameters):
        """Save parameters to config file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(params.to_dict(), f, indent=2)
            logger.debug(f"Parameters saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save parameters: {e}")
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """
        Record performance metrics for auto-tuning
        
        Args:
            metrics: Performance metrics from query processing
        """
        with self.lock:
            self.performance_history.append(metrics)
            self.recent_metrics.append(metrics)
            self.query_count += 1
            
            # Check if optimization is needed
            if self.query_count % self.optimization_interval == 0:
                self.auto_adjust_parameters()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics from recent performance data"""
        if not self.recent_metrics:
            return {}
        
        metrics_list = list(self.recent_metrics)
        
        return {
            'avg_response_time': statistics.mean([m.response_time for m in metrics_list]),
            'avg_quality_score': statistics.mean([m.quality_score for m in metrics_list]),
            'avg_memory_usage': statistics.mean([m.memory_usage_mb for m in metrics_list]),
            'avg_cache_hit_rate': statistics.mean([m.cache_hit_rate for m in metrics_list]),
            'avg_throughput': statistics.mean([m.throughput_queries_per_min for m in metrics_list]),
            'avg_error_rate': statistics.mean([m.error_rate for m in metrics_list]),
            'response_time_std': statistics.stdev([m.response_time for m in metrics_list]) if len(metrics_list) > 1 else 0,
            'quality_score_std': statistics.stdev([m.quality_score for m in metrics_list]) if len(metrics_list) > 1 else 0,
        }
    
    def auto_adjust_parameters(self):
        """Automatically adjust parameters based on performance metrics"""
        metrics = self.calculate_metrics()
        if not metrics:
            return
        
        logger.info(f"Auto-tuning parameters based on {len(self.recent_metrics)} recent queries")
        
        # Store previous parameters for comparison
        self.previous_params = self.current_params.to_dict()
        
        # Adjust parameters based on strategy
        if self.optimization_strategy == OptimizationStrategy.RESPONSE_TIME:
            self._optimize_for_response_time(metrics)
        elif self.optimization_strategy == OptimizationStrategy.QUALITY_SCORE:
            self._optimize_for_quality(metrics)
        elif self.optimization_strategy == OptimizationStrategy.MEMORY_USAGE:
            self._optimize_for_memory(metrics)
        elif self.optimization_strategy == OptimizationStrategy.CACHE_HIT_RATE:
            self._optimize_for_cache(metrics)
        elif self.optimization_strategy == OptimizationStrategy.THROUGHPUT:
            self._optimize_for_throughput(metrics)
        else:  # BALANCED
            self._optimize_balanced(metrics)
        
        # Save updated parameters
        self.save_parameters(self.current_params)
        
        # Check if optimization improved performance
        self._evaluate_optimization(metrics)
        
        self.optimization_count += 1
        self.last_optimization = time.time()
        
        logger.info(f"Auto-tuning completed. Parameters updated: {self._get_changed_params()}")
    
    def _optimize_for_response_time(self, metrics: Dict[str, float]):
        """Optimize for response time"""
        avg_response_time = metrics['avg_response_time']
        
        # Adjust retrieval_k based on response time
        if avg_response_time > self.PERFORMANCE_THRESHOLDS['response_time_slow']:
            # Response time is too slow, reduce retrieval_k
            new_retrieval_k = max(
                self.TUNABLE_PARAMS['retrieval_k'][0],
                self.current_params.retrieval_k - 2
            )
            self.current_params.retrieval_k = new_retrieval_k
            logger.info(f"Reduced retrieval_k to {new_retrieval_k} due to slow response time")
        
        elif avg_response_time < self.PERFORMANCE_THRESHOLDS['response_time_fast']:
            # Response time is fast, can increase retrieval_k for better quality
            if metrics['avg_quality_score'] < self.PERFORMANCE_THRESHOLDS['quality_score_high']:
                new_retrieval_k = min(
                    self.TUNABLE_PARAMS['retrieval_k'][1],
                    self.current_params.retrieval_k + 1
                )
                self.current_params.retrieval_k = new_retrieval_k
                logger.info(f"Increased retrieval_k to {new_retrieval_k} for better quality")
        
        # Adjust batch_size based on response time
        if avg_response_time > self.PERFORMANCE_THRESHOLDS['response_time_slow']:
            new_batch_size = max(
                self.TUNABLE_PARAMS['batch_size'][0],
                self.current_params.batch_size - 5
            )
            self.current_params.batch_size = new_batch_size
            logger.info(f"Reduced batch_size to {new_batch_size} due to slow response time")
    
    def _optimize_for_quality(self, metrics: Dict[str, float]):
        """Optimize for quality score"""
        avg_quality_score = metrics['avg_quality_score']
        
        # Adjust retrieval_k based on quality
        if avg_quality_score < self.PERFORMANCE_THRESHOLDS['quality_score_low']:
            # Quality is too low, increase retrieval_k
            new_retrieval_k = min(
                self.TUNABLE_PARAMS['retrieval_k'][1],
                self.current_params.retrieval_k + 2
            )
            self.current_params.retrieval_k = new_retrieval_k
            logger.info(f"Increased retrieval_k to {new_retrieval_k} due to low quality")
        
        # Adjust similarity_threshold based on quality
        if avg_quality_score < self.PERFORMANCE_THRESHOLDS['quality_score_low']:
            new_threshold = max(
                self.TUNABLE_PARAMS['similarity_threshold'][0],
                self.current_params.similarity_threshold - 0.05
            )
            self.current_params.similarity_threshold = new_threshold
            logger.info(f"Reduced similarity_threshold to {new_threshold:.2f} for better quality")
        
        # Adjust rerank_k based on quality
        if avg_quality_score < self.PERFORMANCE_THRESHOLDS['quality_score_low']:
            new_rerank_k = min(
                self.TUNABLE_PARAMS['rerank_k'][1],
                self.current_params.rerank_k + 1
            )
            self.current_params.rerank_k = new_rerank_k
            logger.info(f"Increased rerank_k to {new_rerank_k} for better quality")
    
    def _optimize_for_memory(self, metrics: Dict[str, float]):
        """Optimize for memory usage"""
        avg_memory_usage = metrics['avg_memory_usage']
        
        if avg_memory_usage > self.PERFORMANCE_THRESHOLDS['memory_usage_high']:
            # Memory usage is too high, reduce parameters
            new_batch_size = max(
                self.TUNABLE_PARAMS['batch_size'][0],
                self.current_params.batch_size - 10
            )
            self.current_params.batch_size = new_batch_size
            logger.info(f"Reduced batch_size to {new_batch_size} due to high memory usage")
            
            new_chunk_size = max(
                self.TUNABLE_PARAMS['chunk_size'][0],
                self.current_params.chunk_size - 100
            )
            self.current_params.chunk_size = new_chunk_size
            logger.info(f"Reduced chunk_size to {new_chunk_size} due to high memory usage")
    
    def _optimize_for_cache(self, metrics: Dict[str, float]):
        """Optimize for cache hit rate"""
        avg_cache_hit_rate = metrics['avg_cache_hit_rate']
        
        if avg_cache_hit_rate < self.PERFORMANCE_THRESHOLDS['cache_hit_rate_low']:
            # Cache hit rate is too low, increase TTL
            new_cache_ttl = min(
                self.TUNABLE_PARAMS['cache_ttl'][1],
                self.current_params.cache_ttl * 2
            )
            self.current_params.cache_ttl = new_cache_ttl
            logger.info(f"Increased cache_ttl to {new_cache_ttl} due to low hit rate")
        
        elif avg_cache_hit_rate > self.PERFORMANCE_THRESHOLDS['cache_hit_rate_high']:
            # Cache hit rate is high, can reduce TTL
            new_cache_ttl = max(
                self.TUNABLE_PARAMS['cache_ttl'][0],
                self.current_params.cache_ttl // 2
            )
            self.current_params.cache_ttl = new_cache_ttl
            logger.info(f"Reduced cache_ttl to {new_cache_ttl} due to high hit rate")
    
    def _optimize_for_throughput(self, metrics: Dict[str, float]):
        """Optimize for throughput"""
        avg_throughput = metrics['avg_throughput']
        
        # Adjust max_concurrent_queries based on throughput
        if avg_throughput < 10:  # Low throughput
            new_max_concurrent = min(
                self.TUNABLE_PARAMS['max_concurrent_queries'][1],
                self.current_params.max_concurrent_queries + 5
            )
            self.current_params.max_concurrent_queries = new_max_concurrent
            logger.info(f"Increased max_concurrent_queries to {new_max_concurrent} for better throughput")
        
        # Adjust batch_size for throughput
        if avg_throughput < 10:
            new_batch_size = min(
                self.TUNABLE_PARAMS['batch_size'][1],
                self.current_params.batch_size + 10
            )
            self.current_params.batch_size = new_batch_size
            logger.info(f"Increased batch_size to {new_batch_size} for better throughput")
    
    def _optimize_balanced(self, metrics: Dict[str, float]):
        """Balanced optimization considering multiple factors"""
        # Optimize for response time if it's the bottleneck
        if metrics['avg_response_time'] > self.PERFORMANCE_THRESHOLDS['response_time_slow']:
            self._optimize_for_response_time(metrics)
        
        # Optimize for quality if it's too low
        elif metrics['avg_quality_score'] < self.PERFORMANCE_THRESHOLDS['quality_score_low']:
            self._optimize_for_quality(metrics)
        
        # Optimize for memory if it's too high
        elif metrics['avg_memory_usage'] > self.PERFORMANCE_THRESHOLDS['memory_usage_high']:
            self._optimize_for_memory(metrics)
        
        # Optimize for cache if hit rate is low
        elif metrics['avg_cache_hit_rate'] < self.PERFORMANCE_THRESHOLDS['cache_hit_rate_low']:
            self._optimize_for_cache(metrics)
        
        # Fine-tune based on overall performance
        else:
            self._fine_tune_parameters(metrics)
    
    def _fine_tune_parameters(self, metrics: Dict[str, float]):
        """Fine-tune parameters for optimal performance"""
        # Adjust retrieval_k based on quality vs response time trade-off
        if (metrics['avg_quality_score'] < 0.7 and 
            metrics['avg_response_time'] < self.PERFORMANCE_THRESHOLDS['response_time_fast']):
            new_retrieval_k = min(
                self.TUNABLE_PARAMS['retrieval_k'][1],
                self.current_params.retrieval_k + 1
            )
            self.current_params.retrieval_k = new_retrieval_k
            logger.info(f"Fine-tuned: increased retrieval_k to {new_retrieval_k}")
        
        # Adjust similarity_threshold based on precision/recall balance
        if metrics['avg_quality_score'] > 0.9:
            new_threshold = min(
                self.TUNABLE_PARAMS['similarity_threshold'][1],
                self.current_params.similarity_threshold + 0.02
            )
            self.current_params.similarity_threshold = new_threshold
            logger.info(f"Fine-tuned: increased similarity_threshold to {new_threshold:.2f}")
    
    def _evaluate_optimization(self, metrics: Dict[str, float]):
        """Evaluate if the optimization improved performance"""
        if len(self.performance_history) < 200:  # Need enough data
            return
        
        # Compare recent performance with previous performance
        recent_metrics = list(self.recent_metrics)[-50:]
        previous_metrics = list(self.performance_history)[-150:-50]
        
        if len(recent_metrics) < 50 or len(previous_metrics) < 50:
            return
        
        # Calculate improvement
        recent_avg_response = statistics.mean([m.response_time for m in recent_metrics])
        previous_avg_response = statistics.mean([m.response_time for m in previous_metrics])
        
        recent_avg_quality = statistics.mean([m.quality_score for m in recent_metrics])
        previous_avg_quality = statistics.mean([m.quality_score for m in previous_metrics])
        
        # Check if optimization improved performance
        response_improved = recent_avg_response < previous_avg_response
        quality_improved = recent_avg_quality > previous_avg_quality
        
        if response_improved or quality_improved:
            self.improvement_count += 1
            logger.info(f"Optimization improved performance: response_improved={response_improved}, quality_improved={quality_improved}")
        else:
            logger.info("Optimization did not improve performance, will try different approach")
    
    def _get_changed_params(self) -> Dict[str, Tuple[Any, Any]]:
        """Get parameters that changed in the last optimization"""
        changed = {}
        current_dict = self.current_params.to_dict()
        
        for key, current_value in current_dict.items():
            if key in self.previous_params and self.previous_params[key] != current_value:
                changed[key] = (self.previous_params[key], current_value)
        
        return changed
    
    def ab_test_parameter(self, param_name: str, value_a: Any, value_b: Any, 
                         duration: int = 100, traffic_split: float = 0.1) -> str:
        """
        A/B test a parameter to find the optimal value
        
        Args:
            param_name: Name of parameter to test
            value_a: First value to test
            value_b: Second value to test
            duration: Number of queries to test
            traffic_split: Percentage of traffic to use for testing
            
        Returns:
            Winner value ('A' or 'B')
        """
        if param_name not in self.TUNABLE_PARAMS:
            raise ValueError(f"Parameter {param_name} is not tunable")
        
        test_id = f"{param_name}_{int(time.time())}"
        
        # Initialize A/B test
        self.ab_tests[test_id] = {
            'param_name': param_name,
            'value_a': value_a,
            'value_b': value_b,
            'duration': duration,
            'traffic_split': traffic_split,
            'start_time': time.time(),
            'metrics_a': [],
            'metrics_b': [],
            'query_count': 0,
            'completed': False
        }
        
        logger.info(f"Started A/B test {test_id}: {param_name} ({value_a} vs {value_b})")
        
        return test_id
    
    def record_ab_test_metric(self, test_id: str, variant: str, metrics: PerformanceMetrics):
        """Record metrics for A/B test"""
        if test_id not in self.ab_tests:
            return
        
        test = self.ab_tests[test_id]
        if test['completed']:
            return
        
        if variant == 'A':
            test['metrics_a'].append(metrics)
        elif variant == 'B':
            test['metrics_b'].append(metrics)
        
        test['query_count'] += 1
        
        # Check if test is complete
        if test['query_count'] >= test['duration']:
            self._complete_ab_test(test_id)
    
    def _complete_ab_test(self, test_id: str):
        """Complete A/B test and determine winner"""
        test = self.ab_tests[test_id]
        
        if len(test['metrics_a']) < 10 or len(test['metrics_b']) < 10:
            logger.warning(f"A/B test {test_id} has insufficient data")
            return
        
        # Calculate metrics for each variant
        metric_a = self._calculate_ab_test_metric(test['metrics_a'])
        metric_b = self._calculate_ab_test_metric(test['metrics_b'])
        
        # Perform statistical significance test
        confidence = self._calculate_statistical_significance(test['metrics_a'], test['metrics_b'])
        
        # Determine winner
        if confidence > 0.95:  # 95% confidence level
            if metric_a > metric_b:
                winner = 'A'
                winning_value = test['value_a']
            else:
                winner = 'B'
                winning_value = test['value_b']
            
            # Apply winning parameter
            setattr(self.current_params, test['param_name'], winning_value)
            self.save_parameters(self.current_params)
            
            logger.info(f"A/B test {test_id} completed: {winner} wins (confidence: {confidence:.2f})")
        else:
            winner = 'TIE'
            winning_value = None
            logger.info(f"A/B test {test_id} completed: No significant difference (confidence: {confidence:.2f})")
        
        # Store result
        result = ABTestResult(
            param_name=test['param_name'],
            value_a=test['value_a'],
            value_b=test['value_b'],
            metric_a=metric_a,
            metric_b=metric_b,
            confidence=confidence,
            winner=winner,
            sample_size=test['query_count'],
            duration=time.time() - test['start_time']
        )
        
        self.ab_test_results.append(result)
        test['completed'] = True
    
    def _calculate_ab_test_metric(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate composite metric for A/B test evaluation"""
        if not metrics_list:
            return 0.0
        
        # Weighted combination of response time and quality
        avg_response_time = statistics.mean([m.response_time for m in metrics_list])
        avg_quality_score = statistics.mean([m.quality_score for m in metrics_list])
        
        # Normalize and combine (lower response time is better, higher quality is better)
        normalized_response = max(0, 1 - avg_response_time / 5.0)  # Normalize to 0-1
        composite_metric = 0.6 * normalized_response + 0.4 * avg_quality_score
        
        return composite_metric
    
    def _calculate_statistical_significance(self, metrics_a: List[PerformanceMetrics], 
                                          metrics_b: List[PerformanceMetrics]) -> float:
        """Calculate statistical significance using t-test"""
        if len(metrics_a) < 2 or len(metrics_b) < 2:
            return 0.0
        
        # Extract response times for t-test
        response_times_a = [m.response_time for m in metrics_a]
        response_times_b = [m.response_time for m in metrics_b]
        
        # Simple t-test implementation
        mean_a = statistics.mean(response_times_a)
        mean_b = statistics.mean(response_times_b)
        
        var_a = statistics.variance(response_times_a)
        var_b = statistics.variance(response_times_b)
        
        n_a = len(response_times_a)
        n_b = len(response_times_b)
        
        # Pooled standard error
        pooled_se = np.sqrt((var_a / n_a) + (var_b / n_b))
        
        if pooled_se == 0:
            return 0.0
        
        # t-statistic
        t_stat = abs(mean_a - mean_b) / pooled_se
        
        # Degrees of freedom
        df = n_a + n_b - 2
        
        # Approximate confidence level (simplified)
        confidence = min(0.99, t_stat / 10.0)  # Simplified approximation
        
        return confidence
    
    def get_current_parameters(self) -> TuningParameters:
        """Get current tuning parameters"""
        return self.current_params
    
    def set_optimization_strategy(self, strategy: OptimizationStrategy):
        """Set optimization strategy"""
        self.optimization_strategy = strategy
        logger.info(f"Optimization strategy set to: {strategy.value}")
    
    def set_performance_targets(self, targets: Dict[str, float]):
        """Set performance targets"""
        self.performance_targets.update(targets)
        logger.info(f"Performance targets updated: {targets}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        metrics = self.calculate_metrics()
        
        return {
            'current_parameters': self.current_params.to_dict(),
            'recent_metrics': metrics,
            'optimization_stats': {
                'total_queries': self.query_count,
                'optimization_count': self.optimization_count,
                'improvement_count': self.improvement_count,
                'last_optimization': self.last_optimization,
                'optimization_interval': self.optimization_interval,
            },
            'ab_test_stats': {
                'active_tests': len([t for t in self.ab_tests.values() if not t['completed']]),
                'completed_tests': len(self.ab_test_results),
                'recent_results': [asdict(r) for r in self.ab_test_results[-5:]]
            },
            'performance_history_size': len(self.performance_history),
            'recent_metrics_size': len(self.recent_metrics),
        }
    
    def reset_parameters(self):
        """Reset parameters to defaults"""
        self.current_params = TuningParameters()
        self.save_parameters(self.current_params)
        logger.info("Parameters reset to defaults")

# Global auto-tuning system instance
_auto_tuner = None

def get_auto_tuner() -> AutoTuningSystem:
    """Get global auto-tuning system instance"""
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = AutoTuningSystem()
    return _auto_tuner

def record_query_metrics(response_time: float, quality_score: float, 
                        memory_usage_mb: float, cache_hit_rate: float,
                        throughput_queries_per_min: float, error_rate: float):
    """Record query metrics for auto-tuning"""
    metrics = PerformanceMetrics(
        response_time=response_time,
        quality_score=quality_score,
        memory_usage_mb=memory_usage_mb,
        cache_hit_rate=cache_hit_rate,
        throughput_queries_per_min=throughput_queries_per_min,
        error_rate=error_rate,
        timestamp=time.time()
    )
    
    auto_tuner = get_auto_tuner()
    auto_tuner.record_metrics(metrics)

def get_tuned_parameters() -> TuningParameters:
    """Get current tuned parameters"""
    auto_tuner = get_auto_tuner()
    return auto_tuner.get_current_parameters()
