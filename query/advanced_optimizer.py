"""
Advanced Query Optimizer with Intelligent Caching and Medical Context Optimization
"""
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from query.enhancer import QueryEnhancer
from core.globals import get_globals

logger = logging.getLogger(__name__)


class QueryCache:
    """Persistent query cache with semantic similarity matching"""
    
    def __init__(self, cache_dir: Path = None, ttl_hours: int = 24):
        """
        Initialize query cache.
        
        Args:
            cache_dir: Directory for cache files (default: ./cache/queries)
            ttl_hours: Time-to-live for cached results in hours
        """
        self.cache_dir = cache_dir or Path("cache/queries")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.embedding_cache_file = self.cache_dir / "embeddings_cache.json"
        self.results_cache_file = self.cache_dir / "results_cache.json"
        
        self.metadata = self._load_metadata()
        self.embedding_cache = self._load_embedding_cache()
        self.results_cache = self._load_results_cache()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'semantic_matches': 0,
            'expired_entries': 0
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {
            'created_at': datetime.now().isoformat(),
            'total_queries': 0,
            'last_cleanup': datetime.now().isoformat()
        }
    
    def _load_embedding_cache(self) -> Dict[str, Any]:
        """Load embedding cache from disk"""
        if self.embedding_cache_file.exists():
            try:
                with open(self.embedding_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
        return {}
    
    def _load_results_cache(self) -> Dict[str, Any]:
        """Load results cache from disk"""
        if self.results_cache_file.exists():
            try:
                with open(self.results_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results cache: {e}")
        return {}
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _save_results_cache(self):
        """Save results cache to disk"""
        try:
            with open(self.results_cache_file, 'w') as f:
                json.dump(self.results_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results cache: {e}")
    
    def get_fingerprint(self, query: str) -> str:
        """Generate SHA256 fingerprint for query"""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()
    
    def is_expired(self, timestamp: str) -> bool:
        """Check if cache entry is expired"""
        try:
            cached_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - cached_time
            return age > timedelta(hours=self.ttl_hours)
        except:
            return True
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        initial_count = len(self.results_cache)
        
        # Clean results cache
        expired_keys = []
        for key, value in self.results_cache.items():
            if self.is_expired(value.get('timestamp', '')):
                expired_keys.append(key)
                self.stats['expired_entries'] += 1
        
        for key in expired_keys:
            del self.results_cache[key]
            # Also remove from embedding cache
            if key in self.embedding_cache:
                del self.embedding_cache[key]
        
        if expired_keys:
            self._save_results_cache()
            self._save_embedding_cache()
            logger.info(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
        
        # Update metadata
        self.metadata['last_cleanup'] = datetime.now().isoformat()
        self._save_metadata()
    
    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Get cached result by fingerprint"""
        if fingerprint in self.results_cache:
            entry = self.results_cache[fingerprint]
            if not self.is_expired(entry.get('timestamp', '')):
                self.stats['hits'] += 1
                logger.debug(f"📎 Cache hit for fingerprint: {fingerprint[:8]}...")
                return entry['result']
            else:
                # Remove expired entry
                del self.results_cache[fingerprint]
                if fingerprint in self.embedding_cache:
                    del self.embedding_cache[fingerprint]
                self.stats['expired_entries'] += 1
        
        self.stats['misses'] += 1
        return None
    
    def set(self, fingerprint: str, query: str, result: Any, embedding: List[float] = None):
        """Store result in cache"""
        self.results_cache[fingerprint] = {
            'query': query,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        if embedding:
            self.embedding_cache[fingerprint] = {
                'embedding': embedding,
                'query': query
            }
        
        # Update metadata
        self.metadata['total_queries'] += 1
        
        # Save to disk
        self._save_results_cache()
        if embedding:
            self._save_embedding_cache()
        self._save_metadata()
        
        logger.debug(f"💾 Cached result for fingerprint: {fingerprint[:8]}...")
    
    def find_similar(self, query_embedding: List[float], threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Find similar cached query using cosine similarity"""
        if not self.embedding_cache:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        # Convert query embedding to numpy array
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        for fingerprint, cache_entry in self.embedding_cache.items():
            # Skip if result is expired
            if fingerprint not in self.results_cache:
                continue
            
            if self.is_expired(self.results_cache[fingerprint].get('timestamp', '')):
                continue
            
            # Calculate cosine similarity
            cached_vec = np.array(cache_entry['embedding']).reshape(1, -1)
            similarity = cosine_similarity(query_vec, cached_vec)[0][0]
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = fingerprint
        
        if best_match:
            self.stats['semantic_matches'] += 1
            logger.info(f"🎯 Found similar cached query (similarity: {best_similarity:.3f})")
            return best_match, best_similarity
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.results_cache),
            'total_requests': total_requests,
            'cache_hits': self.stats['hits'],
            'cache_misses': self.stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'semantic_matches': self.stats['semantic_matches'],
            'expired_cleaned': self.stats['expired_entries']
        }


class QueryOptimizer(QueryEnhancer):
    """
    Advanced Query Optimizer with intelligent caching and medical context optimization.
    Extends QueryEnhancer with multi-level caching and advanced optimization.
    """
    
    def __init__(self, cache_dir: Path = None, cache_ttl_hours: int = 24, similarity_threshold: float = 0.85):
        """
        Initialize QueryOptimizer.
        
        Args:
            cache_dir: Directory for persistent cache
            cache_ttl_hours: Time-to-live for cached results
            similarity_threshold: Threshold for semantic similarity matching
        """
        super().__init__()
        
        # Initialize cache
        self.cache = QueryCache(cache_dir=cache_dir, ttl_hours=cache_ttl_hours)
        self.similarity_threshold = similarity_threshold
        
        # Initialize embeddings model
        self._init_embeddings()
        
        # Metrics tracking
        self.metrics = {
            'total_queries': 0,
            'optimized_queries': 0,
            'cache_hits': 0,
            'semantic_matches': 0,
            'avg_response_time': 0.0,
            'time_saved': 0.0
        }
        
        # Medical intent patterns (extended)
        self.medical_intent_patterns = {
            'diagnosis': {
                'patterns': [r'\bdiagnos', r'\bdetect', r'\bidentif', r'\bscreen', r'\btest'],
                'mesh_terms': ['diagnosis', 'diagnostic techniques', 'early diagnosis', 'differential diagnosis'],
                'temporal_context': 'early OR initial OR differential'
            },
            'treatment': {
                'patterns': [r'\btreat', r'\btherap', r'\bmanag', r'\binterven', r'\bcure'],
                'mesh_terms': ['therapeutics', 'treatment outcome', 'therapy', 'drug therapy'],
                'temporal_context': 'first-line OR second-line OR adjuvant OR combination'
            },
            'mechanism': {
                'patterns': [r'\bmechanism', r'\bpathway', r'\bpathophysiolog', r'\bcause', r'\betiology'],
                'mesh_terms': ['molecular mechanisms', 'pathophysiology', 'signal transduction', 'etiology'],
                'temporal_context': None
            },
            'epidemiology': {
                'patterns': [r'\bepidemiolog', r'\bprevalence', r'\bincidence', r'\brisk factor', r'\bdistribution'],
                'mesh_terms': ['epidemiology', 'prevalence', 'incidence', 'risk factors', 'disease outbreaks'],
                'temporal_context': 'recent OR current OR trends OR global'
            },
            'prognosis': {
                'patterns': [r'\bprognos', r'\boutcome', r'\bsurviv', r'\bmortalit', r'\brecurrence'],
                'mesh_terms': ['prognosis', 'survival analysis', 'treatment outcome', 'mortality'],
                'temporal_context': '5-year OR 10-year OR long-term OR short-term'
            },
            'prevention': {
                'patterns': [r'\bprevent', r'\bprophyla', r'\bvaccin', r'\bimmuniz', r'\bprotect'],
                'mesh_terms': ['prevention and control', 'primary prevention', 'secondary prevention', 'vaccines'],
                'temporal_context': 'primary OR secondary OR tertiary'
            }
        }
        
        # Common query patterns for pre-computation
        self.common_patterns = [
            "COVID-19 treatment",
            "diabetes management",
            "cancer screening",
            "hypertension diagnosis",
            "depression therapy",
            "vaccine efficacy",
            "antibiotic resistance",
            "heart disease prevention",
            "stroke rehabilitation",
            "chronic pain management"
        ]
        
        # Pre-compute embeddings for common patterns
        self._precompute_common_patterns()
        
        # Cleanup expired entries periodically
        self.cache.cleanup_expired()
        
        logger.info("✅ QueryOptimizer initialized with intelligent caching")
    
    def _init_embeddings(self):
        """Initialize embeddings model for semantic similarity"""
        try:
            globals_dict = get_globals()
            self.embeddings = globals_dict.get('embeddings')
            
            if not self.embeddings:
                # Fallback to a simple embedding model
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("📊 Initialized fallback embeddings model")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None
    
    @lru_cache(maxsize=1000)
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text with LRU caching"""
        if not self.embeddings:
            return None
        
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _precompute_common_patterns(self):
        """Pre-compute embeddings for common query patterns"""
        logger.info("🔄 Pre-computing embeddings for common patterns...")
        
        for pattern in self.common_patterns:
            # Generate embedding
            embedding = self._generate_embedding(pattern)
            
            if embedding:
                # Store in cache
                fingerprint = self.cache.get_fingerprint(pattern)
                
                # Create mock result for common patterns
                enhanced = self.enhance_query(pattern)
                result = {
                    'original_query': pattern,
                    'optimized_query': enhanced,
                    'enhanced_query': enhanced,
                    'precomputed': True,
                    'pattern': pattern,
                    'intents': self.detect_medical_intent(pattern),
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': 0.001,
                    'optimizations_applied': {
                        'acronyms_expanded': False,
                        'mesh_terms_added': True,
                        'temporal_context': False,
                        'boolean_optimized': True
                    }
                }
                
                self.cache.set(fingerprint, pattern, result, embedding)
        
        logger.info(f"✅ Pre-computed {len(self.common_patterns)} common patterns")
    
    def detect_medical_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect detailed medical intent from query.
        
        Args:
            query: Input query string
            
        Returns:
            Dictionary with detected intents and confidence scores
        """
        query_lower = query.lower()
        detected_intents = {}
        
        for intent, config in self.medical_intent_patterns.items():
            score = 0.0
            matched_patterns = []
            
            for pattern in config['patterns']:
                if re.search(pattern, query_lower):
                    score += 1.0
                    matched_patterns.append(pattern)
            
            if score > 0:
                # Normalize score
                score = score / len(config['patterns'])
                detected_intents[intent] = {
                    'confidence': score,
                    'matched_patterns': matched_patterns,
                    'mesh_terms': config['mesh_terms'],
                    'temporal_context': config.get('temporal_context')
                }
        
        return detected_intents
    
    def optimize_for_medical_context(self, query: str) -> str:
        """
        Optimize query for medical context with advanced features.
        
        Args:
            query: Input query string
            
        Returns:
            Optimized query string
        """
        start_time = time.time()
        
        # Detect medical intent
        intents = self.detect_medical_intent(query)
        
        # Start with basic enhancement (from parent class)
        enhanced_query = self.expand_acronyms(query)
        
        # Add MeSH terms based on intent
        mesh_additions = []
        temporal_additions = []
        
        for intent, details in intents.items():
            if details['confidence'] > 0.3:  # Only add if confident
                mesh_additions.extend(details['mesh_terms'][:2])  # Limit to top 2 terms
                
                if details['temporal_context']:
                    temporal_additions.append(details['temporal_context'])
        
        # Build optimized query
        query_parts = [enhanced_query]
        
        if mesh_additions:
            mesh_part = " OR ".join([f'"{term}"[MeSH]' for term in mesh_additions])
            query_parts.append(f"({mesh_part})")
        
        if temporal_additions:
            temporal_part = " OR ".join(temporal_additions)
            query_parts.append(f"({temporal_part})")
        
        # Add time-sensitive filters for certain intents
        if 'epidemiology' in intents or 'treatment' in intents:
            # Add recent publication filter
            query_parts.append('("last 5 years"[PDat])')
        
        # Optimize boolean logic
        optimized_query = self._optimize_boolean_logic(" AND ".join(query_parts))
        
        # Track metrics
        elapsed = time.time() - start_time
        self.metrics['optimized_queries'] += 1
        
        logger.info(f"🔧 Optimized query in {elapsed:.3f}s with intents: {list(intents.keys())}")
        
        return optimized_query
    
    def _optimize_boolean_logic(self, query: str) -> str:
        """
        Simplify and optimize boolean logic in query.
        
        Args:
            query: Query with boolean operators
            
        Returns:
            Optimized query
        """
        # Remove redundant parentheses
        query = re.sub(r'\(\(([^)]+)\)\)', r'(\1)', query)
        
        # Remove duplicate terms
        terms = set()
        parts = query.split(' OR ')
        unique_parts = []
        
        for part in parts:
            cleaned = part.strip().lower()
            if cleaned not in terms:
                terms.add(cleaned)
                unique_parts.append(part)
        
        if len(unique_parts) < len(parts):
            query = ' OR '.join(unique_parts)
            logger.debug(f"Removed {len(parts) - len(unique_parts)} duplicate terms")
        
        return query
    
    def find_similar_cached_query(self, query: str, threshold: float = None) -> Optional[Dict[str, Any]]:
        """
        Find similar cached query using semantic similarity.
        
        Args:
            query: Input query
            threshold: Similarity threshold (default: self.similarity_threshold)
            
        Returns:
            Cached result if found, None otherwise
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        
        if not query_embedding:
            return None
        
        # Find similar query in cache
        match = self.cache.find_similar(query_embedding, threshold)
        
        if match:
            fingerprint, similarity = match
            result = self.cache.get(fingerprint)
            
            if result:
                self.metrics['semantic_matches'] += 1
                logger.info(f"✨ Using cached result from similar query (similarity: {similarity:.3f})")
                return result
        
        return None
    
    def optimize_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Main optimization method with caching and all features.
        
        Args:
            query: Input query
            use_cache: Whether to use caching
            
        Returns:
            Dictionary with optimized query and metadata
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Check exact match in cache
        fingerprint = self.cache.get_fingerprint(query)
        
        if use_cache:
            # Try exact match first
            cached_result = self.cache.get(fingerprint)
            if cached_result:
                self.metrics['cache_hits'] += 1
                elapsed = time.time() - start_time
                self.metrics['time_saved'] += (0.5 - elapsed)  # Assume 0.5s for full processing
                logger.info(f"⚡ Cache hit! Retrieved in {elapsed:.3f}s")
                return cached_result
            
            # Try semantic similarity match
            similar_result = self.find_similar_cached_query(query)
            if similar_result:
                elapsed = time.time() - start_time
                self.metrics['time_saved'] += (0.5 - elapsed)
                return similar_result
        
        # No cache hit - perform full optimization
        logger.info(f"🔍 Processing new query: {query}")
        
        # Detect intents
        intents = self.detect_medical_intent(query)
        
        # Optimize for medical context
        optimized_query = self.optimize_for_medical_context(query)
        
        # Add temporal context for time-sensitive queries
        if any(term in query.lower() for term in ['recent', 'latest', 'current', 'new']):
            optimized_query += ' AND ("last 2 years"[PDat])'
        
        # Create result
        result = {
            'original_query': query,
            'optimized_query': optimized_query,
            'intents': intents,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time,
            'optimizations_applied': {
                'acronyms_expanded': query != self.expand_acronyms(query),
                'mesh_terms_added': bool(intents),
                'temporal_context': 'recent' in query.lower() or 'latest' in query.lower(),
                'boolean_optimized': True
            }
        }
        
        # Cache the result
        if use_cache:
            query_embedding = self._generate_embedding(query)
            self.cache.set(fingerprint, query, result, query_embedding)
        
        # Update metrics
        elapsed = time.time() - start_time
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (self.metrics['total_queries'] - 1) + elapsed) 
            / self.metrics['total_queries']
        )
        
        logger.info(f"✅ Query optimized in {elapsed:.3f}s")
        
        return result
    
    def analyze_query_logs(self, log_file: Path = None) -> Dict[str, Any]:
        """
        Analyze query logs to identify patterns for pre-computation.
        
        Args:
            log_file: Path to query log file
            
        Returns:
            Analysis results with common patterns
        """
        if not log_file or not log_file.exists():
            logger.warning("No query log file provided or file doesn't exist")
            return {}
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            # Extract query patterns
            query_counts = {}
            for entry in logs:
                query = entry.get('query', '').lower().strip()
                if query:
                    query_counts[query] = query_counts.get(query, 0) + 1
            
            # Sort by frequency
            top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:100]
            
            # Pre-compute embeddings for top queries
            logger.info(f"📊 Pre-computing embeddings for top {len(top_queries)} queries")
            
            for query, count in top_queries:
                if count > 5:  # Only cache frequently used queries
                    self.optimize_query(query, use_cache=True)
            
            return {
                'total_unique_queries': len(query_counts),
                'top_queries': top_queries[:20],
                'patterns_cached': len(top_queries)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze query logs: {e}")
            return {}
    
    def warm_cache(self):
        """Warm cache with common medical queries on startup"""
        logger.info("🔥 Warming cache with common medical queries...")
        
        common_queries = [
            "COVID-19 vaccine efficacy",
            "diabetes type 2 treatment guidelines",
            "hypertension first line therapy",
            "breast cancer screening recommendations",
            "depression cognitive behavioral therapy",
            "antibiotic resistance mechanisms",
            "heart failure management guidelines",
            "stroke prevention strategies",
            "chronic pain non-pharmacological treatment",
            "pneumonia diagnosis criteria",
            "asthma inhaler therapy",
            "migraine prophylaxis",
            "osteoporosis prevention",
            "COPD exacerbation treatment",
            "atrial fibrillation anticoagulation"
        ]
        
        for query in common_queries:
            try:
                self.optimize_query(query, use_cache=True)
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query}': {e}")
        
        logger.info(f"✅ Cache warmed with {len(common_queries)} common queries")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics"""
        cache_stats = self.cache.get_stats()
        
        return {
            'optimizer_metrics': {
                'total_queries': self.metrics['total_queries'],
                'optimized_queries': self.metrics['optimized_queries'],
                'cache_hits': self.metrics['cache_hits'],
                'semantic_matches': self.metrics['semantic_matches'],
                'avg_response_time_ms': self.metrics['avg_response_time'] * 1000,
                'total_time_saved_s': self.metrics['time_saved']
            },
            'cache_stats': cache_stats,
            'performance_improvement': {
                'response_time_reduction': f"{(self.metrics['time_saved'] / max(self.metrics['total_queries'], 1)) * 100:.1f}%",
                'cache_effectiveness': cache_stats['hit_rate']
            }
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.results_cache.clear()
        self.cache.embedding_cache.clear()
        self.cache._save_results_cache()
        self.cache._save_embedding_cache()
        self._generate_embedding.cache_clear()
        logger.info("🗑️ Cache cleared")


# Integration wrapper for chains.py
class OptimizedQueryWrapper:
    """Wrapper to integrate QueryOptimizer with existing chains"""
    
    def __init__(self, optimizer: QueryOptimizer = None):
        """
        Initialize wrapper with optimizer.
        
        Args:
            optimizer: QueryOptimizer instance (creates new if None)
        """
        self.optimizer = optimizer or QueryOptimizer()
        self.original_enhancer = QueryEnhancer()
        
        # Track usage
        self.usage_stats = {
            'queries_processed': 0,
            'optimization_time': 0.0,
            'cache_hits': 0
        }
    
    def enhance_query(self, query: str) -> str:
        """
        Enhanced query method compatible with existing interface.
        
        Args:
            query: Input query
            
        Returns:
            Optimized query string
        """
        start_time = time.time()
        
        # Use optimizer
        result = self.optimizer.optimize_query(query, use_cache=True)
        
        # Update stats
        self.usage_stats['queries_processed'] += 1
        self.usage_stats['optimization_time'] += (time.time() - start_time)
        
        if result.get('cached', False):
            self.usage_stats['cache_hits'] += 1
        
        # Return optimized query (maintains compatibility)
        return result.get('optimized_query', query)
    
    def get_detailed_result(self, query: str) -> Dict[str, Any]:
        """
        Get detailed optimization result.
        
        Args:
            query: Input query
            
        Returns:
            Full optimization result with metadata
        """
        return self.optimizer.optimize_query(query, use_cache=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'wrapper_stats': self.usage_stats,
            'optimizer_metrics': self.optimizer.get_metrics()
        }


# Singleton instance for global use
_global_optimizer = None

def get_query_optimizer() -> QueryOptimizer:
    """Get or create global QueryOptimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = QueryOptimizer()
        _global_optimizer.warm_cache()
    return _global_optimizer


def integrate_with_chains():
    """
    Integrate QueryOptimizer with existing chains.py.
    This function should be called during application initialization.
    """
    try:
        from utils import chains
        
        # Get global optimizer
        optimizer = get_query_optimizer()
        wrapper = OptimizedQueryWrapper(optimizer)
        
        # Replace QueryEnhancer in chains.py with optimized version
        if hasattr(chains, 'get_or_create_chain'):
            # Monkey-patch the QueryEnhancer with our optimizer
            original_get_or_create = chains.get_or_create_chain
            
            def optimized_get_or_create_chain(topic_id: str, conversation_id: str, query: str):
                # Log the transformation
                logger.info(f"🚀 Using QueryOptimizer for query: {query[:50]}...")
                
                # Get optimization result
                opt_result = optimizer.optimize_query(query, use_cache=True)
                
                # Log optimization details
                if opt_result.get('optimizations_applied'):
                    applied = [k for k, v in opt_result['optimizations_applied'].items() if v]
                    logger.info(f"✅ Applied optimizations: {', '.join(applied)}")
                
                # Call original with optimized query
                return original_get_or_create(topic_id, conversation_id, opt_result['optimized_query'])
            
            # Replace function
            chains.get_or_create_chain = optimized_get_or_create_chain
            
            # Also update the singleton _query_enhancer if it exists
            if hasattr(chains.get_or_create_chain, '_query_enhancer'):
                chains.get_or_create_chain._query_enhancer = wrapper
            
            logger.info("✅ QueryOptimizer integrated with chains.py")
            return True
            
    except Exception as e:
        logger.error(f"Failed to integrate QueryOptimizer: {e}")
        return False
