"""
Graph-Based Document Retriever for Medical Literature
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import networkx as nx

from .builder import MedicalKnowledgeGraph

logger = logging.getLogger(__name__)

class GraphRetriever:
    """Retrieve documents using knowledge graph and semantic similarity"""
    
    def __init__(self, knowledge_graph: MedicalKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Graph centrality cache
        self.centrality_cache = {}
        self._compute_centrality()
    
    def _compute_centrality(self):
        """Compute centrality measures for graph nodes"""
        if not self.knowledge_graph.graph:
            return
        
        try:
            # Compute different centrality measures
            degree_centrality = nx.degree_centrality(self.knowledge_graph.graph)
            betweenness_centrality = nx.betweenness_centrality(self.knowledge_graph.graph)
            closeness_centrality = nx.closeness_centrality(self.knowledge_graph.graph)
            
            # Combine centrality measures
            for node in self.knowledge_graph.graph.nodes():
                self.centrality_cache[node] = {
                    'degree': degree_centrality.get(node, 0.0),
                    'betweenness': betweenness_centrality.get(node, 0.0),
                    'closeness': closeness_centrality.get(node, 0.0),
                    'combined': (degree_centrality.get(node, 0.0) + 
                               betweenness_centrality.get(node, 0.0) + 
                               closeness_centrality.get(node, 0.0)) / 3
                }
            
            logger.info(f"Computed centrality for {len(self.centrality_cache)} nodes")
            
        except Exception as e:
            logger.warning(f"Error computing centrality: {e}")
    
    def graph_search(self, query: str, documents: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """Search documents using knowledge graph and semantic similarity"""
        if not self.knowledge_graph.graph or not documents:
            return documents[:k]
        
        try:
            # Extract entities from query
            query_entities = self.knowledge_graph.entity_extractor.extract_entities(query)
            
            # Score documents based on graph relevance
            scored_documents = []
            
            for doc in documents:
                score = self._calculate_graph_score(query, query_entities, doc)
                doc_copy = doc.copy()
                doc_copy['graph_score'] = score
                scored_documents.append(doc_copy)
            
            # Sort by graph score
            scored_documents.sort(key=lambda x: x.get('graph_score', 0), reverse=True)
            
            logger.info(f"Graph search completed: {len(scored_documents)} documents scored")
            return scored_documents[:k]
            
        except Exception as e:
            logger.error(f"Error in graph search: {e}")
            return documents[:k]
    
    def _calculate_graph_score(self, query: str, query_entities: Dict[str, List[str]], document: Dict[str, Any]) -> float:
        """Calculate graph-based relevance score for a document"""
        score = 0.0
        
        # Extract entities from document
        doc_title = document.get('title', '')
        doc_abstract = document.get('abstract', '')
        doc_content = f"{doc_title}. {doc_abstract}"
        
        doc_entities = self.knowledge_graph.entity_extractor.extract_entities(doc_content)
        
        # Entity overlap score
        entity_overlap = self._calculate_entity_overlap(query_entities, doc_entities)
        score += entity_overlap * 0.4
        
        # Graph centrality score
        centrality_score = self._calculate_centrality_score(doc_entities)
        score += centrality_score * 0.3
        
        # Semantic similarity score
        semantic_score = self._calculate_semantic_similarity(query, doc_content)
        score += semantic_score * 0.3
        
        return score
    
    def _calculate_entity_overlap(self, query_entities: Dict[str, List[str]], doc_entities: Dict[str, List[str]]) -> float:
        """Calculate overlap between query and document entities"""
        if not query_entities or not doc_entities:
            return 0.0
        
        total_query_entities = sum(len(entities) for entities in query_entities.values())
        if total_query_entities == 0:
            return 0.0
        
        overlap_count = 0
        
        for entity_type, query_entity_list in query_entities.items():
            if entity_type in doc_entities:
                doc_entity_list = doc_entities[entity_type]
                for query_entity in query_entity_list:
                    for doc_entity in doc_entity_list:
                        if query_entity.lower() == doc_entity.lower():
                            overlap_count += 1
        
        return overlap_count / total_query_entities
    
    def _calculate_centrality_score(self, doc_entities: Dict[str, List[str]]) -> float:
        """Calculate centrality score based on document entities"""
        if not doc_entities or not self.centrality_cache:
            return 0.0
        
        centrality_scores = []
        
        for entity_type, entity_list in doc_entities.items():
            for entity in entity_list:
                if entity in self.centrality_cache:
                    centrality_scores.append(self.centrality_cache[entity]['combined'])
        
        if centrality_scores:
            return np.mean(centrality_scores)
        
        return 0.0
    
    def _calculate_semantic_similarity(self, query: str, document_content: str) -> float:
        """Calculate semantic similarity between query and document"""
        try:
            # Encode query and document
            query_embedding = self.semantic_model.encode([query])[0]
            doc_embedding = self.semantic_model.encode([document_content])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def get_related_entities(self, entity: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities related to a given entity through the knowledge graph"""
        if not self.knowledge_graph.graph:
            return []
        
        neighbors = self.knowledge_graph.get_entity_neighbors(entity, max_depth)
        
        related_entities = []
        for neighbor, data in neighbors.items():
            related_entities.append({
                'entity': neighbor,
                'type': data['type'],
                'depth': data['depth'],
                'relationships': data['relationships'],
                'centrality': self.centrality_cache.get(neighbor, {}).get('combined', 0.0)
            })
        
        # Sort by centrality
        related_entities.sort(key=lambda x: x['centrality'], reverse=True)
        return related_entities
    
    def find_entity_paths(self, source_entity: str, target_entity: str, max_paths: int = 5) -> List[List[Dict[str, Any]]]:
        """Find paths between two entities in the knowledge graph"""
        if not self.knowledge_graph.graph:
            return []
        
        return self.knowledge_graph.get_relationship_path(source_entity, target_entity, max_paths)
    
    def expand_query_with_graph(self, query: str) -> Dict[str, Any]:
        """Expand query using knowledge graph entities and relationships"""
        if not self.knowledge_graph.graph:
            return {'original_query': query, 'expanded_query': query, 'entities': {}}
        
        # Extract entities from query
        query_entities = self.knowledge_graph.entity_extractor.extract_entities(query)
        
        # Find related entities
        related_entities = {}
        for entity_type, entity_list in query_entities.items():
            for entity in entity_list:
                related = self.get_related_entities(entity, max_depth=1)
                if related:
                    related_entities[entity] = related[:3]  # Top 3 related entities
        
        # Build expanded query
        expanded_terms = []
        for entity_type, entity_list in query_entities.items():
            expanded_terms.extend(entity_list)
            
            # Add related entities
            for entity in entity_list:
                if entity in related_entities:
                    for related in related_entities[entity]:
                        expanded_terms.append(related['entity'])
        
        # Remove duplicates and build expanded query
        unique_terms = list(set(expanded_terms))
        expanded_query = " OR ".join([f'"{term}"' for term in unique_terms])
        
        return {
            'original_query': query,
            'expanded_query': expanded_query,
            'entities': query_entities,
            'related_entities': related_entities
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return self.knowledge_graph.get_graph_statistics()
