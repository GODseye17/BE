"""
Performance Monitoring for tracking metrics
"""
import time
import psutil
import logging
from typing import Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def track_performance(self, operation: str):
        """Decorator to track operation performance"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise e
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    
                    duration = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    self._record_metric(operation, duration, memory_used, success)
                
                return result
            return wrapper
        return decorator
    
    def _record_metric(self, operation: str, duration: float, memory: int, success: bool):
        """Record performance metric"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append({
            'duration': duration,
            'memory': memory,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 metrics
        if len(self.metrics[operation]) > 1000:
            self.metrics[operation] = self.metrics[operation][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        result = {}
        for operation, metrics in self.metrics.items():
            if metrics:
                durations = [m['duration'] for m in metrics]
                memories = [m['memory'] for m in metrics]
                success_rate = sum(1 for m in metrics if m['success']) / len(metrics)
                
                result[operation] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'avg_memory': sum(memories) / len(memories),
                    'success_rate': success_rate,
                    'total_operations': len(metrics)
                }
        
        return result
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available': psutil.virtual_memory().available,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
