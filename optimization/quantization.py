"""
Embedding Quantization for Memory and Storage Optimization

This module implements embedding quantization techniques to reduce memory usage
and storage requirements while maintaining reasonable quality.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
import time
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for embedding quantization"""
    method: str = "int8"  # "int8", "fp16", "pca"
    compression_ratio: float = 4.0  # Target compression ratio
    quality_threshold: float = 0.95  # Minimum quality preservation
    pca_components: Optional[int] = None  # Number of PCA components
    enable_cache: bool = True
    cache_size: int = 1000

@dataclass
class QuantizationResult:
    """Result of quantization operation"""
    original_shape: Tuple[int, int]
    quantized_shape: Tuple[int, int]
    compression_ratio: float
    quality_score: float
    memory_saved_mb: float
    quantization_time: float
    method: str
    metadata: Dict[str, Any]

class EmbeddingQuantizer:
    """
    Embedding quantizer for memory and storage optimization
    
    Supports multiple quantization methods:
    - int8: 8-bit integer quantization (4x compression)
    - fp16: 16-bit floating point (2x compression)
    - pca: Principal Component Analysis (variable compression)
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize embedding quantizer
        
        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance tracking
        self.total_quantizations = 0
        self.total_memory_saved = 0.0
        self.avg_compression_ratio = 0.0
        self.avg_quality_score = 0.0
        
        logger.info(f"EmbeddingQuantizer initialized with method: {self.config.method}")
    
    def quantize_embeddings(self, embeddings: np.ndarray, 
                           method: Optional[str] = None) -> Tuple[np.ndarray, QuantizationResult]:
        """
        Quantize embeddings using specified method
        
        Args:
            embeddings: Input embeddings array
            method: Quantization method (overrides config)
            
        Returns:
            Tuple of (quantized_embeddings, quantization_result)
        """
        method = method or self.config.method
        
        # Check cache first
        cache_key = self._get_cache_key(embeddings, method)
        with self.cache_lock:
            if cache_key in self.cache and self.config.enable_cache:
                logger.debug("Using cached quantization result")
                return self.cache[cache_key]
        
        start_time = time.time()
        original_shape = embeddings.shape
        original_memory = embeddings.nbytes / (1024 * 1024)  # MB
        
        logger.info(f"Quantizing embeddings: {original_shape} using {method}")
        
        try:
            if method == "int8":
                quantized_embeddings, metadata = self._quantize_int8(embeddings)
            elif method == "fp16":
                quantized_embeddings, metadata = self._quantize_fp16(embeddings)
            elif method == "pca":
                quantized_embeddings, metadata = self._quantize_pca(embeddings)
            else:
                raise ValueError(f"Unknown quantization method: {method}")
            
            # Calculate results
            quantized_memory = quantized_embeddings.nbytes / (1024 * 1024)  # MB
            compression_ratio = original_memory / quantized_memory
            memory_saved = original_memory - quantized_memory
            quality_score = self._calculate_quality_score(embeddings, quantized_embeddings)
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                original_shape=original_shape,
                quantized_shape=quantized_embeddings.shape,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                memory_saved_mb=memory_saved,
                quantization_time=quantization_time,
                method=method,
                metadata=metadata
            )
            
            # Cache result
            with self.cache_lock:
                if self.config.enable_cache:
                    self.cache[cache_key] = (quantized_embeddings, result)
                    # Limit cache size
                    if len(self.cache) > self.config.cache_size:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
            
            # Update statistics
            self._update_statistics(result)
            
            logger.info(f"Quantization completed: {compression_ratio:.2f}x compression, "
                       f"{quality_score:.3f} quality, {memory_saved:.2f}MB saved")
            
            return quantized_embeddings, result
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            # Return original embeddings if quantization fails
            return embeddings, QuantizationResult(
                original_shape=original_shape,
                quantized_shape=original_shape,
                compression_ratio=1.0,
                quality_score=1.0,
                memory_saved_mb=0.0,
                quantization_time=time.time() - start_time,
                method="failed",
                metadata={"error": str(e)}
            )
    
    def _quantize_int8(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to 8-bit integers"""
        # Calculate scaling factors
        min_val = np.min(embeddings)
        max_val = np.max(embeddings)
        
        # Avoid division by zero
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 255.0
            zero_point = int(round(-min_val / scale))
        
        # Quantize
        quantized = np.round((embeddings - min_val) / scale).astype(np.int8)
        
        metadata = {
            "min_val": min_val,
            "max_val": max_val,
            "scale": scale,
            "zero_point": zero_point,
            "dtype": "int8"
        }
        
        return quantized, metadata
    
    def _quantize_fp16(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize to 16-bit floating point"""
        # Convert to float16
        quantized = embeddings.astype(np.float16)
        
        metadata = {
            "original_dtype": str(embeddings.dtype),
            "dtype": "float16"
        }
        
        return quantized, metadata
    
    def _quantize_pca(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize using Principal Component Analysis"""
        try:
            from sklearn.decomposition import PCA
            
            # Determine number of components
            if self.config.pca_components is None:
                # Auto-determine based on compression ratio
                target_components = int(embeddings.shape[1] / self.config.compression_ratio)
                n_components = max(1, min(target_components, embeddings.shape[1] - 1))
            else:
                n_components = min(self.config.pca_components, embeddings.shape[1] - 1)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            quantized = pca.fit_transform(embeddings)
            
            # Store PCA components for reconstruction
            metadata = {
                "pca_components": pca.components_,
                "pca_mean": pca.mean_,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "n_components": n_components,
                "dtype": "pca"
            }
            
            return quantized, metadata
            
        except ImportError:
            logger.warning("sklearn not available, falling back to simple dimensionality reduction")
            # Simple fallback: take first n components
            n_components = max(1, int(embeddings.shape[1] / self.config.compression_ratio))
            quantized = embeddings[:, :n_components]
            
            metadata = {
                "n_components": n_components,
                "method": "simple_reduction",
                "dtype": "reduced"
            }
            
            return quantized, metadata
    
    def dequantize_embeddings(self, quantized_embeddings: np.ndarray, 
                             metadata: Dict[str, Any]) -> np.ndarray:
        """
        Dequantize embeddings back to original format
        
        Args:
            quantized_embeddings: Quantized embeddings
            metadata: Quantization metadata
            
        Returns:
            Dequantized embeddings
        """
        method = metadata.get("dtype", "unknown")
        
        try:
            if method == "int8":
                return self._dequantize_int8(quantized_embeddings, metadata)
            elif method == "float16":
                return self._dequantize_fp16(quantized_embeddings, metadata)
            elif method == "pca":
                return self._dequantize_pca(quantized_embeddings, metadata)
            else:
                logger.warning(f"Unknown dequantization method: {method}")
                return quantized_embeddings
                
        except Exception as e:
            logger.error(f"Dequantization failed: {e}")
            return quantized_embeddings
    
    def _dequantize_int8(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Dequantize from 8-bit integers"""
        min_val = metadata["min_val"]
        scale = metadata["scale"]
        
        return quantized.astype(np.float32) * scale + min_val
    
    def _dequantize_fp16(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Dequantize from 16-bit floating point"""
        return quantized.astype(np.float32)
    
    def _dequantize_pca(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Dequantize from PCA representation"""
        try:
            from sklearn.decomposition import PCA
            
            pca_components = metadata["pca_components"]
            pca_mean = metadata["pca_mean"]
            
            # Reconstruct original embeddings
            reconstructed = np.dot(quantized, pca_components) + pca_mean
            
            return reconstructed
            
        except ImportError:
            logger.warning("sklearn not available for PCA dequantization")
            return quantized
    
    def _calculate_quality_score(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """Calculate quality preservation score"""
        try:
            # Calculate cosine similarity between original and quantized
            original_norm = original / np.linalg.norm(original, axis=1, keepdims=True)
            quantized_norm = quantized / np.linalg.norm(quantized, axis=1, keepdims=True)
            
            # Calculate cosine similarity for each embedding
            similarities = np.sum(original_norm * quantized_norm, axis=1)
            
            # Return average similarity
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.5  # Default quality score
    
    def _get_cache_key(self, embeddings: np.ndarray, method: str) -> str:
        """Generate cache key for embeddings"""
        # Use shape and method as cache key
        return f"{embeddings.shape}_{method}_{embeddings.nbytes}"
    
    def _update_statistics(self, result: QuantizationResult):
        """Update performance statistics"""
        self.total_quantizations += 1
        
        # Update running averages
        self.total_memory_saved += result.memory_saved_mb
        self.avg_compression_ratio = (
            (self.avg_compression_ratio * (self.total_quantizations - 1) + result.compression_ratio) 
            / self.total_quantizations
        )
        self.avg_quality_score = (
            (self.avg_quality_score * (self.total_quantizations - 1) + result.quality_score) 
            / self.total_quantizations
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quantization statistics"""
        return {
            'total_quantizations': self.total_quantizations,
            'total_memory_saved_mb': self.total_memory_saved,
            'avg_compression_ratio': self.avg_compression_ratio,
            'avg_quality_score': self.avg_quality_score,
            'cache_size': len(self.cache),
            'config': {
                'method': self.config.method,
                'compression_ratio': self.config.compression_ratio,
                'quality_threshold': self.config.quality_threshold,
                'enable_cache': self.config.enable_cache
            }
        }
    
    def clear_cache(self):
        """Clear quantization cache"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("Quantization cache cleared")
    
    def save_quantized_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, Any], 
                                 filepath: str):
        """Save quantized embeddings to file"""
        try:
            data = {
                'embeddings': embeddings,
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Quantized embeddings saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save quantized embeddings: {e}")
    
    def load_quantized_embeddings(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load quantized embeddings from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            logger.info(f"Quantized embeddings loaded from {filepath}")
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"Failed to load quantized embeddings: {e}")
            raise

class QuantizedVectorStore:
    """Vector store with quantized embeddings for memory efficiency"""
    
    def __init__(self, quantizer: Optional[EmbeddingQuantizer] = None):
        """
        Initialize quantized vector store
        
        Args:
            quantizer: Embedding quantizer instance
        """
        self.quantizer = quantizer or EmbeddingQuantizer()
        self.embeddings = None
        self.quantized_embeddings = None
        self.metadata = None
        self.documents = []
        
        logger.info("QuantizedVectorStore initialized")
    
    def add_embeddings(self, embeddings: np.ndarray, documents: List[Any], 
                      quantize: bool = True) -> QuantizationResult:
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: Input embeddings
            documents: Associated documents
            quantize: Whether to quantize embeddings
            
        Returns:
            Quantization result
        """
        self.embeddings = embeddings
        self.documents = documents
        
        if quantize:
            self.quantized_embeddings, result = self.quantizer.quantize_embeddings(embeddings)
            self.metadata = result.metadata
            return result
        else:
            self.quantized_embeddings = embeddings
            self.metadata = {"dtype": "original"}
            return QuantizationResult(
                original_shape=embeddings.shape,
                quantized_shape=embeddings.shape,
                compression_ratio=1.0,
                quality_score=1.0,
                memory_saved_mb=0.0,
                quantization_time=0.0,
                method="none",
                metadata=self.metadata
            )
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Any, float]]:
        """
        Perform similarity search using quantized embeddings
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.quantized_embeddings is None:
            raise ValueError("No embeddings available for search")
        
        # Quantize query embedding if needed
        if self.metadata.get("dtype") in ["int8", "pca"]:
            query_quantized, _ = self.quantizer.quantize_embeddings(
                query_embedding.reshape(1, -1), 
                method=self.metadata.get("dtype")
            )
            query_quantized = query_quantized.flatten()
        else:
            query_quantized = query_embedding
        
        # Calculate similarities
        similarities = np.dot(self.quantized_embeddings, query_quantized)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if self.embeddings is None:
            return {"original_mb": 0.0, "quantized_mb": 0.0, "savings_mb": 0.0}
        
        original_mb = self.embeddings.nbytes / (1024 * 1024)
        quantized_mb = self.quantized_embeddings.nbytes / (1024 * 1024)
        savings_mb = original_mb - quantized_mb
        
        return {
            "original_mb": original_mb,
            "quantized_mb": quantized_mb,
            "savings_mb": savings_mb,
            "compression_ratio": original_mb / quantized_mb if quantized_mb > 0 else 1.0
        }

# Global quantizer instance
_global_quantizer = None

def get_quantizer() -> EmbeddingQuantizer:
    """Get global quantizer instance"""
    global _global_quantizer
    if _global_quantizer is None:
        _global_quantizer = EmbeddingQuantizer()
    return _global_quantizer
