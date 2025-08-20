"""
Article Relevance Scoring System
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class ArticleRelevanceScorer:
    """Comprehensive relevance scoring for PubMed articles"""
    
    def __init__(self):
        # Initialize sentence transformer for semantic similarity
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded semantic similarity model")
        except Exception as e:
            logger.warning(f"⚠️ Could not load semantic model: {e}")
            self.semantic_model = None
        
        # Weights for different scoring components
        self.weights = {
            'title_exact_match': 0.3,
            'semantic_similarity': 0.25,
            'keyword_coverage': 0.2,
            'mesh_relevance': 0.15,
            'recency': 0.1
        }
        
        # Penalty factors
        self.penalties = {
            'short_abstract': 0.1,
            'missing_authors': 0.05,
            'low_quality_pub_type': 0.15
        }
    
    def calculate_title_exact_match_score(self, title: str, query_terms: List[str]) -> float:
        """Calculate exact match score for query terms in title"""
        if not title or not query_terms:
            return 0.0
        
        title_lower = title.lower()
        matched_terms = 0
        
        for term in query_terms:
            term_lower = term.lower()
            # Check for exact word match (not substring)
            if re.search(r'\b' + re.escape(term_lower) + r'\b', title_lower):
                matched_terms += 1
        
        return matched_terms / len(query_terms) if query_terms else 0.0
    
    def calculate_semantic_similarity_score(self, text: str, query: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if not self.semantic_model or not text or not query:
            return 0.0
        
        try:
            # Encode text and query
            text_embedding = self.semantic_model.encode([text])[0]
            query_embedding = self.semantic_model.encode([query])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(text_embedding, query_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(query_embedding)
            )
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_keyword_coverage_score(self, title: str, abstract: str, query_terms: List[str]) -> float:
        """Calculate keyword coverage in title and abstract"""
        if not query_terms:
            return 0.0
        
        combined_text = f"{title} {abstract}".lower()
        matched_terms = 0
        
        for term in query_terms:
            term_lower = term.lower()
            # Check for word match in combined text
            if re.search(r'\b' + re.escape(term_lower) + r'\b', combined_text):
                matched_terms += 1
        
        return matched_terms / len(query_terms) if query_terms else 0.0
    
    def calculate_mesh_relevance_score(self, mesh_terms: List[str], query_terms: List[str]) -> float:
        """Calculate MeSH term relevance score"""
        if not mesh_terms or not query_terms:
            return 0.0
        
        mesh_text = " ".join(mesh_terms).lower()
        matched_terms = 0
        
        for term in query_terms:
            term_lower = term.lower()
            # Check for term match in MeSH terms
            if re.search(r'\b' + re.escape(term_lower) + r'\b', mesh_text):
                matched_terms += 1
        
        return matched_terms / len(query_terms) if query_terms else 0.0
    
    def calculate_recency_score(self, publication_date: str) -> float:
        """Calculate recency score based on publication date"""
        if not publication_date or publication_date == 'Unknown Date':
            return 0.5  # Neutral score for unknown dates
        
        try:
            # Extract year from date
            year_str = publication_date.split('-')[0]
            if not year_str.isdigit():
                return 0.5
            
            year = int(year_str)
            current_year = datetime.now().year
            
            # Calculate years since publication
            years_old = current_year - year
            
            # Score based on recency (recent articles score higher)
            if years_old <= 2:
                return 1.0
            elif years_old <= 5:
                return 0.9
            elif years_old <= 10:
                return 0.7
            elif years_old <= 15:
                return 0.5
            elif years_old <= 20:
                return 0.3
            else:
                return 0.1
                
        except (ValueError, IndexError):
            return 0.5
    
    def calculate_penalties(self, article_data: Dict[str, Any]) -> float:
        """Calculate penalty score based on article quality issues"""
        penalty = 0.0
        
        # Penalty for short abstract
        abstract = article_data.get('abstract', '')
        if len(abstract.strip()) < 200:
            penalty += self.penalties['short_abstract']
        
        # Penalty for missing authors
        authors = article_data.get('authors', '')
        if not authors or authors == 'Unknown Authors':
            penalty += self.penalties['missing_authors']
        
        # Penalty for low-quality publication types
        pub_types = article_data.get('publication_types', [])
        low_quality_types = ['Editorial', 'Letter', 'Comment', 'News']
        if any(low_quality_type in pub_types for low_quality_type in low_quality_types):
            penalty += self.penalties['low_quality_pub_type']
        
        return penalty
    
    def extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query"""
        # Remove common stop words and punctuation
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Clean and split query
        query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = query_clean.split()
        
        # Filter out stop words and short terms
        meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        return meaningful_terms
    
    def calculate_relevance_score(self, article_data: Dict[str, Any], query: str) -> float:
        """Calculate comprehensive relevance score for an article"""
        # Extract query terms
        query_terms = self.extract_query_terms(query)
        
        # Get article data
        title = article_data.get('title', '')
        abstract = article_data.get('abstract', '')
        mesh_terms = article_data.get('mesh_terms', [])
        publication_date = article_data.get('publication_date', '')
        
        # Calculate individual scores
        title_score = self.calculate_title_exact_match_score(title, query_terms)
        semantic_score = self.calculate_semantic_similarity_score(f"{title} {abstract}", query)
        keyword_score = self.calculate_keyword_coverage_score(title, abstract, query_terms)
        mesh_score = self.calculate_mesh_relevance_score(mesh_terms, query_terms)
        recency_score = self.calculate_recency_score(publication_date)
        
        # Calculate weighted score
        weighted_score = (
            title_score * self.weights['title_exact_match'] +
            semantic_score * self.weights['semantic_similarity'] +
            keyword_score * self.weights['keyword_coverage'] +
            mesh_score * self.weights['mesh_relevance'] +
            recency_score * self.weights['recency']
        )
        
        # Apply penalties
        penalty = self.calculate_penalties(article_data)
        final_score = max(0.0, weighted_score - penalty)
        
        # Store individual scores in metadata for debugging
        article_data['relevance_scores'] = {
            'title_exact_match': title_score,
            'semantic_similarity': semantic_score,
            'keyword_coverage': keyword_score,
            'mesh_relevance': mesh_score,
            'recency': recency_score,
            'penalty': penalty,
            'final_score': final_score
        }
        
        return final_score
    
    def filter_articles_by_relevance(self, articles_data: List[Dict[str, Any]], 
                                   query_info: Dict[str, Any], 
                                   min_score: float = 0.4) -> List[Dict[str, Any]]:
        """Filter articles based on relevance score"""
        query = query_info.get('original_query', '')
        key_terms = query_info.get('key_terms', [])
        
        if not query:
            logger.warning("No query provided for relevance filtering")
            return articles_data
        
        logger.info(f"🔍 Calculating relevance scores for {len(articles_data)} articles")
        
        scored_articles = []
        for article in articles_data:
            try:
                relevance_score = self.calculate_relevance_score(article, query)
                article['relevance_score'] = relevance_score
                
                if relevance_score >= min_score:
                    scored_articles.append(article)
                    logger.debug(f"✅ Article {article.get('pmid', 'unknown')} scored {relevance_score:.3f}")
                else:
                    logger.debug(f"❌ Article {article.get('pmid', 'unknown')} scored {relevance_score:.3f} (below threshold)")
                    
            except Exception as e:
                logger.warning(f"Error scoring article {article.get('pmid', 'unknown')}: {e}")
                # Include article with neutral score if scoring fails
                article['relevance_score'] = 0.5
                scored_articles.append(article)
        
        # Sort by relevance score (highest first)
        scored_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"✅ Filtered to {len(scored_articles)} relevant articles (min_score={min_score})")
        
        return scored_articles
