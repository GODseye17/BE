"""
Query Enhancement for PubMed Searches with Advanced Optimization and Caching
"""
import logging
import re
import hashlib
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

class QueryEnhancer:
    """Enhance queries with medical acronyms, synonyms, and optimization"""
    
    def __init__(self):
        # Medical acronyms dictionary
        self.medical_acronyms = {
            'mi': 'myocardial infarction',
            'copd': 'chronic obstructive pulmonary disease',
            'cvd': 'cardiovascular disease',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'cva': 'cerebrovascular accident',
            'tia': 'transient ischemic attack',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'uti': 'urinary tract infection',
            'pneumonia': 'pneumonia',
            'sepsis': 'sepsis',
            'ards': 'acute respiratory distress syndrome',
            'aki': 'acute kidney injury',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'cirrhosis': 'cirrhosis',
            'hepatitis': 'hepatitis',
            'ibd': 'inflammatory bowel disease',
            'uc': 'ulcerative colitis',
            'cd': 'crohn disease',
            'ra': 'rheumatoid arthritis',
            'oa': 'osteoarthritis',
            'lupus': 'systemic lupus erythematosus',
            'sle': 'systemic lupus erythematosus',
            'ms': 'multiple sclerosis',
            'pd': 'parkinson disease',
            'ad': 'alzheimer disease',
            'dementia': 'dementia',
            'depression': 'depression',
            'anxiety': 'anxiety',
            'ptsd': 'post traumatic stress disorder',
            'ocd': 'obsessive compulsive disorder',
            'adhd': 'attention deficit hyperactivity disorder',
            'autism': 'autism spectrum disorder',
            'asd': 'autism spectrum disorder',
            'cancer': 'neoplasms',
            'tumor': 'neoplasms',
            'metastasis': 'neoplasm metastasis',
            'chemotherapy': 'drug therapy',
            'radiation': 'radiotherapy',
            'surgery': 'surgical procedures operative',
            'transplant': 'transplantation',
            'vaccine': 'vaccines',
            'antibiotic': 'anti-bacterial agents',
            'antiviral': 'antiviral agents',
            'immunotherapy': 'immunotherapy',
            'targeted therapy': 'molecular targeted therapy'
        }
        
        # Medical synonyms dictionary
        self.medical_synonyms = {
            'heart attack': 'myocardial infarction',
            'stroke': 'cerebrovascular accident',
            'high blood pressure': 'hypertension',
            'diabetes': 'diabetes mellitus',
            'cancer': 'neoplasms',
            'tumor': 'neoplasms',
            'kidney disease': 'kidney diseases',
            'liver disease': 'liver diseases',
            'lung disease': 'lung diseases',
            'heart disease': 'heart diseases',
            'brain disease': 'brain diseases',
            'mental illness': 'mental disorders',
            'psychiatric disorder': 'mental disorders',
            'drug': 'pharmaceutical preparations',
            'medicine': 'pharmaceutical preparations',
            'treatment': 'therapy',
            'therapy': 'therapy',
            'diagnosis': 'diagnosis',
            'symptoms': 'signs and symptoms',
            'side effects': 'adverse effects',
            'complications': 'complications',
            'risk factors': 'risk factors',
            'prevention': 'prevention and control',
            'screening': 'mass screening',
            'early detection': 'early diagnosis',
            'prognosis': 'prognosis',
            'survival': 'survival',
            'mortality': 'mortality',
            'morbidity': 'morbidity',
            'quality of life': 'quality of life',
            'patient outcomes': 'treatment outcome',
            'clinical trial': 'clinical trial',
            'randomized trial': 'randomized controlled trial',
            'meta analysis': 'meta-analysis',
            'systematic review': 'systematic review',
            'cohort study': 'cohort studies',
            'case control': 'case-control studies',
            'observational study': 'observational study',
            'epidemiology': 'epidemiology',
            'prevalence': 'prevalence',
            'incidence': 'incidence',
            'mortality rate': 'mortality',
            'survival rate': 'survival',
            'response rate': 'treatment outcome',
            'remission': 'remission induction',
            'relapse': 'neoplasm recurrence local',
            'recurrence': 'neoplasm recurrence local'
        }
        
        # Intent detection patterns
        self.intent_patterns = {
            'mechanism': [
                r'\bhow\b', r'\bmechanism\b', r'\bpathway\b', r'\bprocess\b',
                r'\bwhat causes\b', r'\bwhy\b', r'\bunderlying\b'
            ],
            'treatment': [
                r'\btreatment\b', r'\btherapy\b', r'\bintervention\b', r'\bmanagement\b',
                r'\bhow to treat\b', r'\bmedication\b', r'\bdrug\b', r'\bsurgery\b'
            ],
            'diagnosis': [
                r'\bdiagnosis\b', r'\bdetection\b', r'\bscreening\b', r'\btest\b',
                r'\bhow to diagnose\b', r'\bsymptoms\b', r'\bsigns\b'
            ],
            'prevention': [
                r'\bprevention\b', r'\bprevent\b', r'\brisk factors\b', r'\bprotective\b',
                r'\bhow to prevent\b', r'\bavoid\b'
            ],
            'comparison': [
                r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference\b',
                r'\bsimilarities\b', r'\bcontrast\b', r'\bbetter\b', r'\bworse\b'
            ],
            'comprehensive': [
                r'\ball\b', r'\bevery\b', r'\bcomprehensive\b', r'\boverview\b',
                r'\bsummary\b', r'\breview\b', r'\bmeta\b'
            ]
        }
    
    def expand_acronyms(self, query: str) -> str:
        """Expand medical acronyms in the query"""
        if not query:
            return query
        
        # Convert to lowercase for matching
        query_lower = query.lower()
        expanded_query = query
        
        # Replace acronyms with their full forms
        for acronym, full_form in self.medical_acronyms.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query_lower):
                expanded_query = re.sub(pattern, full_form, expanded_query, flags=re.IGNORECASE)
                logger.debug(f"Expanded acronym: {acronym} -> {full_form}")
        
        return expanded_query
    
    def add_synonyms(self, query: str) -> str:
        """Add medical synonyms to the query"""
        if not query:
            return query
        
        query_lower = query.lower()
        enhanced_query = query
        
        # Add synonyms using OR operator
        synonyms_to_add = []
        for synonym, medical_term in self.medical_synonyms.items():
            if re.search(r'\b' + re.escape(synonym) + r'\b', query_lower):
                synonyms_to_add.append(medical_term)
                logger.debug(f"Added synonym: {synonym} -> {medical_term}")
        
        # Add synonyms to query
        if synonyms_to_add:
            synonym_part = " OR ".join([f'"{term}"' for term in synonyms_to_add])
            enhanced_query = f"({enhanced_query}) OR ({synonym_part})"
        
        return enhanced_query
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Detect query intent for optimization"""
        query_lower = query.lower()
        detected_intents = {}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents[intent] = True
                    break
        
        return detected_intents
    
    def _build_optimal_query(self, query: str, intents: Dict[str, Any]) -> str:
        """Build optimized PubMed query based on detected intents"""
        # Start with the enhanced query
        optimized_query = query
        
        # Add MeSH terms based on intent
        mesh_additions = []
        
        if intents.get('mechanism'):
            mesh_additions.extend(['"molecular mechanisms"', '"pathophysiology"'])
        
        if intents.get('treatment'):
            mesh_additions.extend(['"therapy"', '"treatment outcome"'])
        
        if intents.get('diagnosis'):
            mesh_additions.extend(['"diagnosis"', '"diagnostic techniques"'])
        
        if intents.get('prevention'):
            mesh_additions.extend(['"prevention and control"', '"risk factors"'])
        
        if intents.get('comparison'):
            mesh_additions.extend(['"comparative study"', '"comparative effectiveness"'])
        
        if intents.get('comprehensive'):
            mesh_additions.extend(['"systematic review"', '"meta-analysis"'])
        
        # Add MeSH terms to query
        if mesh_additions:
            mesh_part = " OR ".join(mesh_additions)
            optimized_query = f"({optimized_query}) AND ({mesh_part})"
        
        return optimized_query
    
    def enhance_query(self, query: str) -> str:
        """Main method to enhance query with all improvements - returns enhanced query string"""
        if not query:
            return query
        
        logger.info(f"🔧 Enhancing query: {query}")
        
        # Step 1: Expand acronyms
        expanded_query = self.expand_acronyms(query)
        
        # Step 2: Add synonyms
        synonym_query = self.add_synonyms(expanded_query)
        
        # Step 3: Detect intent
        intents = self.detect_intent(query)
        
        # Step 4: Build optimized query
        optimized_query = self._build_optimal_query(synonym_query, intents)
        
        # Add field restrictions for better precision
        if not re.search(r'\[Title\]|\[Title/Abstract\]', optimized_query):
            # Add title boosting
            optimized_query = f'({optimized_query}[Title]) OR ({optimized_query}[Title/Abstract])'
        
        logger.info(f"✅ Enhanced query: {optimized_query}")
        logger.info(f"📊 Detected intents: {list(intents.keys())}")
        
        return optimized_query
    
    def enhance_query_detailed(self, query: str) -> Dict[str, Any]:
        """Enhanced query method that returns detailed information"""
        if not query:
            return {'original_query': query, 'enhanced_query': query, 'intents': {}}
        
        # Step 1: Expand acronyms
        expanded_query = self.expand_acronyms(query)
        
        # Step 2: Add synonyms
        synonym_query = self.add_synonyms(expanded_query)
        
        # Step 3: Detect intent
        intents = self.detect_intent(query)
        
        # Step 4: Build optimized query
        optimized_query = self._build_optimal_query(synonym_query, intents)
        
        # Add field restrictions for better precision
        if not re.search(r'\[Title\]|\[Title/Abstract\]', optimized_query):
            # Add title boosting
            optimized_query = f'({optimized_query}[Title]) OR ({optimized_query}[Title/Abstract])'
        
        result = {
            'original_query': query,
            'enhanced_query': optimized_query,
            'intents': intents,
            'expansions': {
                'acronyms_expanded': expanded_query != query,
                'synonyms_added': synonym_query != expanded_query,
                'mesh_terms_added': optimized_query != synonym_query
            }
        }
        
        return result


class QueryOptimizer:
    """Advanced Query Optimizer with LRU Cache and Semantic Similarity Matching"""
    
    def __init__(self, cache_size: int = 1000, similarity_threshold: float = 0.85, cache_ttl: int = 86400):
        """
        Initialize QueryOptimizer
        
        Args:
            cache_size: Maximum number of cached queries
            similarity_threshold: Semantic similarity threshold for cache hits
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        """
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        
        # LRU cache for query results
        self.query_cache = OrderedDict()
        self.cache_metadata = {}  # Store timestamps and fingerprints
        
        # Initialize query enhancer
        self.query_enhancer = QueryEnhancer()
        
        # Semantic similarity cache
        self.similarity_cache = {}
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_optimizations = 0
        
        logger.info(f"🧠 QueryOptimizer initialized: cache_size={cache_size}, threshold={similarity_threshold}")
    
    def _get_query_fingerprint(self, query: str) -> str:
        """Generate unique fingerprint for query"""
        # Normalize query for fingerprinting
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate semantic similarity between two queries
        Uses simple word overlap for now, can be enhanced with embeddings
        """
        # Create cache key
        cache_key = f"{self._get_query_fingerprint(query1)}_{self._get_query_fingerprint(query2)}"
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Simple word-based similarity (can be enhanced with sentence embeddings)
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            similarity = 0.0
        else:
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            similarity = intersection / union if union > 0 else 0.0
        
        # Cache the result
        self.similarity_cache[cache_key] = similarity
        
        # Limit similarity cache size
        if len(self.similarity_cache) > 10000:
            # Remove oldest entries
            keys_to_remove = list(self.similarity_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.similarity_cache[key]
        
        return similarity
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self.cache_ttl
    
    def _find_similar_cached_query(self, query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Find semantically similar cached query"""
        query_fingerprint = self._get_query_fingerprint(query)
        
        # First try exact match
        if query_fingerprint in self.cache_metadata:
            metadata = self.cache_metadata[query_fingerprint]
            if self._is_cache_valid(metadata['timestamp']):
                if query_fingerprint in self.query_cache:
                    # Move to end (LRU update)
                    result = self.query_cache.pop(query_fingerprint)
                    self.query_cache[query_fingerprint] = result
                    return query_fingerprint, result
        
        # Try semantic similarity matching
        for cached_fingerprint, cached_result in self.query_cache.items():
            metadata = self.cache_metadata.get(cached_fingerprint, {})
            
            # Skip if cache entry is expired
            if not self._is_cache_valid(metadata.get('timestamp', 0)):
                continue
            
            # Calculate similarity with original query
            original_query = metadata.get('original_query', '')
            similarity = self._calculate_semantic_similarity(query, original_query)
            
            if similarity >= self.similarity_threshold:
                # Move to end (LRU update)
                result = self.query_cache.pop(cached_fingerprint)
                self.query_cache[cached_fingerprint] = result
                
                # Update result to indicate cache hit
                result = result.copy()
                result['cached'] = True
                result['cache_similarity'] = similarity
                result['cache_hit_type'] = 'semantic'
                
                logger.info(f"🎯 Semantic cache hit: similarity={similarity:.3f} for query: {query[:50]}...")
                return cached_fingerprint, result
        
        return None
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.cache_metadata.items():
            if current_time - metadata['timestamp'] > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.query_cache:
                del self.query_cache[key]
            del self.cache_metadata[key]
        
        if expired_keys:
            logger.info(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
    
    def _cache_result(self, query: str, result: Dict[str, Any]):
        """Cache optimization result with LRU eviction"""
        query_fingerprint = self._get_query_fingerprint(query)
        
        # Remove oldest entries if cache is full
        while len(self.query_cache) >= self.cache_size:
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
            if oldest_key in self.cache_metadata:
                del self.cache_metadata[oldest_key]
        
        # Add to cache
        self.query_cache[query_fingerprint] = result
        self.cache_metadata[query_fingerprint] = {
            'timestamp': time.time(),
            'original_query': query,
            'fingerprint': query_fingerprint
        }
        
        logger.debug(f"💾 Cached optimization result for: {query[:50]}...")
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Optimize query with caching and semantic similarity
        
        Args:
            query: Input query to optimize
            
        Returns:
            Dictionary with optimized query and metadata
        """
        if not query:
            return {'optimized_query': query, 'cached': False}
        
        self.total_optimizations += 1
        
        # Clean up expired cache entries periodically
        if self.total_optimizations % 100 == 0:
            self._cleanup_expired_cache()
        
        logger.info(f"🔧 Optimizing query: {query}")
        
        # Try to find similar cached query
        cached_result = self._find_similar_cached_query(query)
        if cached_result:
            self.cache_hits += 1
            fingerprint, result = cached_result
            logger.info(f"🎯 Cache hit for query: {query[:50]}...")
            return result
        
        # Cache miss - perform optimization
        self.cache_misses += 1
        logger.info(f"❌ Cache miss, performing optimization: {query[:50]}...")
        
        start_time = time.time()
        
        # Use query enhancer for optimization
        enhanced_result = self.query_enhancer.enhance_query_detailed(query)
        
        optimization_time = time.time() - start_time
        
        # Create comprehensive result
        result = {
            'optimized_query': enhanced_result['enhanced_query'],
            'original_query': query,
            'intents': enhanced_result['intents'],
            'expansions': enhanced_result['expansions'],
            'optimization_time': optimization_time,
            'cached': False,
            'cache_hit_type': 'none',
            'fingerprint': self._get_query_fingerprint(query)
        }
        
        # Cache the result
        self._cache_result(query, result)
        
        logger.info(f"✅ Query optimized in {optimization_time:.3f}s: {query[:50]}...")
        return result
    
    def enhance_query(self, query: str) -> str:
        """
        Enhanced query method that returns just the optimized query string
        (for compatibility with existing code)
        """
        result = self.optimize_query(query)
        return result.get('optimized_query', query)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        return {
            'cache_size': len(self.query_cache),
            'max_cache_size': self.cache_size,
            'total_optimizations': self.total_optimizations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'similarity_threshold': self.similarity_threshold,
            'cache_ttl': self.cache_ttl,
            'similarity_cache_size': len(self.similarity_cache)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.query_cache.clear()
        self.cache_metadata.clear()
        self.similarity_cache.clear()
        logger.info("🗑️ All caches cleared")
    
    def set_similarity_threshold(self, threshold: float):
        """Update similarity threshold"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"🎯 Similarity threshold updated to {threshold}")
        else:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    
    def preload_common_queries(self, queries: List[str]):
        """Preload optimization results for common queries"""
        logger.info(f"🚀 Preloading {len(queries)} common queries...")
        
        for query in queries:
            if query and query not in [meta['original_query'] for meta in self.cache_metadata.values()]:
                self.optimize_query(query)
        
        logger.info(f"✅ Preloaded {len(queries)} queries into cache")


# Global optimizer instance
_global_optimizer = None

def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = QueryOptimizer()
    return _global_optimizer

# Backward compatibility function
@lru_cache(maxsize=128)
def enhance_query_cached(query: str) -> str:
    """Simple cached query enhancement (for backward compatibility)"""
    enhancer = QueryEnhancer()
    return enhancer.enhance_query(query)
