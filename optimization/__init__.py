"""
Optimization Package

This package contains optimization modules for the Vivum Backend,
including auto-tuning systems, performance optimization tools, and quantization.
"""

from .auto_tuner import (
    AutoTuningSystem,
    PerformanceMetrics,
    TuningParameters,
    ABTestResult,
    OptimizationStrategy,
    get_auto_tuner,
    record_query_metrics,
    get_tuned_parameters
)

from .quantization import (
    Float8Quantizer,
    QuantizationStats,
    get_quantizer,
    compress_embeddings,
    decompress_embeddings,
    dequantize_for_search
)

__all__ = [
    'AutoTuningSystem',
    'PerformanceMetrics', 
    'TuningParameters',
    'ABTestResult',
    'OptimizationStrategy',
    'get_auto_tuner',
    'record_query_metrics',
    'get_tuned_parameters',
    'Float8Quantizer',
    'QuantizationStats',
    'get_quantizer',
    'compress_embeddings',
    'decompress_embeddings',
    'dequantize_for_search'
]
