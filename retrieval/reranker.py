"""
Document Reranking with Cross-Encoder Models
"""
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)

class RelevanceReranker:
    """Rerank documents using cross-encoder models for better relevance"""
    
    def __init__(self):
        try:
            # Initialize cross-encoder model for reranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ Loaded cross-encoder reranking model")
        except Exception as e:
            logger.warning(f"⚠️ Could not load cross-encoder model: {e}")
            self.cross_encoder = None
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], 
                        top_k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder model"""
        if not self.cross_encoder or not documents:
            logger.warning("No cross-encoder model available or no documents to rerank")
            return documents[:top_k]
        
        try:
            # Prepare document pairs for cross-encoder
            document_pairs = []
            for doc in documents:
                # Use title and abstract for reranking
                content = doc.get('page_content', '')
                if not content:
                    # Fallback to metadata if content is empty
                    title = doc.get('metadata', {}).get('title', '')
                    abstract = doc.get('metadata', {}).get('abstract', '')
                    content = f"{title} {abstract}".strip()
                
                document_pairs.append([query, content])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(document_pairs)
            
            # Combine documents with scores
            scored_documents = []
            for i, doc in enumerate(documents):
                score = float(scores[i])
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = score
                scored_documents.append(doc_copy)
            
            # Filter by score threshold and sort by score
            filtered_documents = [
                doc for doc in scored_documents 
                if doc['rerank_score'] >= score_threshold
            ]
            
            # Sort by rerank score (highest first)
            filtered_documents.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top_k documents
            result = filtered_documents[:top_k]
            
            logger.info(f"🔍 Reranked {len(documents)} documents -> {len(result)} relevant (threshold={score_threshold})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original documents if reranking fails
            return documents[:top_k]
    
    def get_rerank_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Get rerank scores for documents without filtering"""
        if not self.cross_encoder or not documents:
            return [0.0] * len(documents)
        
        try:
            # Prepare document pairs
            document_pairs = []
            for doc in documents:
                content = doc.get('page_content', '')
                if not content:
                    title = doc.get('metadata', {}).get('title', '')
                    abstract = doc.get('metadata', {}).get('abstract', '')
                    content = f"{title} {abstract}".strip()
                
                document_pairs.append([query, content])
            
            # Get scores
            scores = self.cross_encoder.predict(document_pairs)
            return [float(score) for score in scores]
            
        except Exception as e:
            logger.error(f"Error getting rerank scores: {e}")
            return [0.0] * len(documents)
