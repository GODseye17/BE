"""
Medical Knowledge Graph Builder
"""
import logging
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from pathlib import Path

from .entity_extractor import MedicalEntityExtractor
from .relationship_extractor import RelationshipExtractor

logger = logging.getLogger(__name__)

class MedicalKnowledgeGraph:
    """Build and manage medical knowledge graph from literature"""
    
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.graph = nx.MultiDiGraph()
        
        # Graph metadata
        self.graph_metadata = {
            'total_entities': 0,
            'total_relationships': 0,
            'entity_types': {},
            'relationship_types': {},
            'sources': []
        }
        
        # Cache for centrality calculations
        self.centrality_cache = {}
    
    def __del__(self):
        """Cleanup to prevent memory leaks"""
        if hasattr(self, 'graph'):
            self.graph.clear()
        if hasattr(self, 'centrality_cache'):
            self.centrality_cache.clear()
    
    def build_from_articles(self, articles: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """Build knowledge graph from a list of articles"""
        logger.info(f"Building knowledge graph from {len(articles)} articles")
        
        for i, article in enumerate(articles):
            try:
                logger.debug(f"Processing article {i+1}/{len(articles)}: {article.get('pmid', 'unknown')}")
                
                # Extract text content
                title = article.get('title', '')
                abstract = article.get('abstract', '')
                text_content = f"{title}. {abstract}"
                
                # Extract entities
                entities = self.entity_extractor.extract_entities(text_content)
                
                # Extract relationships
                relationships = self.relationship_extractor.extract_relationships(text_content, entities)
                
                # Add to graph
                self._add_entities_to_graph(entities, article.get('pmid', f'article_{i}'))
                self._add_relationships_to_graph(relationships, article.get('pmid', f'article_{i}'))
                
                # Update metadata
                self._update_metadata(entities, relationships, article.get('pmid', f'article_{i}'))
                
            except Exception as e:
                logger.warning(f"Error processing article {i}: {e}")
                continue
        
        logger.info(f"Knowledge graph built with {self.graph_metadata['total_entities']} entities and {self.graph_metadata['total_relationships']} relationships")
        return self.graph
    
    def _add_entities_to_graph(self, entities: Dict[str, List[str]], source: str):
        """Add entities to the knowledge graph"""
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # Add node if it doesn't exist
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, 
                                      type=entity_type,
                                      sources=[source],
                                      first_seen=source)
                else:
                    # Update existing node
                    node_data = self.graph.nodes[entity]
                    if source not in node_data.get('sources', []):
                        node_data['sources'].append(source)
    
    def _add_relationships_to_graph(self, relationships: List[Dict[str, Any]], source: str):
        """Add relationships to the knowledge graph"""
        for relationship in relationships:
            source_entity = relationship['source']
            target_entity = relationship['target']
            rel_type = relationship['relationship']
            
            # Validate relationship
            if not self.relationship_extractor.validate_relationship(relationship):
                logger.debug(f"Skipping invalid relationship: {source_entity} -> {target_entity}")
                continue
            
            # Add edge
            edge_data = {
                'relationship_type': rel_type,
                'confidence': relationship['confidence'],
                'context': relationship['context'],
                'source_type': relationship['source_type'],
                'target_type': relationship['target_type'],
                'sources': [source]
            }
            
            self.graph.add_edge(source_entity, target_entity, **edge_data)
    
    def _update_metadata(self, entities: Dict[str, List[str]], relationships: List[Dict[str, Any]], source: str):
        """Update graph metadata"""
        # Update entity counts
        for entity_type, entity_list in entities.items():
            if entity_type not in self.graph_metadata['entity_types']:
                self.graph_metadata['entity_types'][entity_type] = 0
            self.graph_metadata['entity_types'][entity_type] += len(entity_list)
        
        # Update relationship counts
        for relationship in relationships:
            rel_type = relationship['relationship']
            if rel_type not in self.graph_metadata['relationship_types']:
                self.graph_metadata['relationship_types'][rel_type] = 0
            self.graph_metadata['relationship_types'][rel_type] += 1
        
        # Update total counts
        self.graph_metadata['total_entities'] = len(self.graph.nodes)
        self.graph_metadata['total_relationships'] = len(self.graph.edges)
        
        # Add source
        if source not in self.graph_metadata['sources']:
            self.graph_metadata['sources'].append(source)
    
    def get_entity_neighbors(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get neighboring entities up to a certain depth"""
        if not self.graph.has_node(entity):
            return {}
        
        neighbors = {}
        visited = set()
        queue = [(entity, 0)]  # (node, depth)
        
        while queue:
            current_entity, depth = queue.pop(0)
            
            if depth > max_depth or current_entity in visited:
                continue
            
            visited.add(current_entity)
            
            if depth > 0:  # Don't include the original entity
                neighbors[current_entity] = {
                    'depth': depth,
                    'type': self.graph.nodes[current_entity].get('type', 'UNKNOWN'),
                    'relationships': []
                }
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(current_entity):
                if neighbor not in visited:
                    # Get relationship information
                    edge_data = self.graph.get_edge_data(current_entity, neighbor)
                    if edge_data:
                        for edge_key, data in edge_data.items():
                            neighbors[current_entity]['relationships'].append({
                                'target': neighbor,
                                'type': data.get('relationship_type', 'unknown'),
                                'confidence': data.get('confidence', 0.0)
                            })
                    
                    queue.append((neighbor, depth + 1))
        
        return neighbors
    
    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for entities in the graph"""
        results = []
        query_lower = query.lower()
        
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            
            # Filter by entity type if specified
            if entity_type and node_data.get('type') != entity_type:
                continue
            
            # Check if query matches entity name
            if query_lower in node.lower():
                results.append({
                    'entity': node,
                    'type': node_data.get('type', 'UNKNOWN'),
                    'sources': node_data.get('sources', []),
                    'neighbor_count': len(list(self.graph.neighbors(node))),
                    'edge_count': len(list(self.graph.edges(node)))
                })
        
        # Sort by relevance (neighbor count as proxy)
        results.sort(key=lambda x: x['neighbor_count'], reverse=True)
        return results
    
    def get_relationship_path(self, source: str, target: str, max_paths: int = 5) -> List[List[Dict[str, Any]]]:
        """Find paths between two entities"""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []
        
        try:
            # Find all simple paths
            all_paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=4))
            
            # Convert paths to detailed format
            detailed_paths = []
            for path in all_paths[:max_paths]:
                detailed_path = []
                
                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i + 1]
                    
                    # Get edge data
                    edge_data = self.graph.get_edge_data(current_node, next_node)
                    if edge_data:
                        for edge_key, data in edge_data.items():
                            detailed_path.append({
                                'source': current_node,
                                'target': next_node,
                                'relationship': data.get('relationship_type', 'unknown'),
                                'confidence': data.get('confidence', 0.0),
                                'context': data.get('context', '')
                            })
                            break  # Take first edge if multiple exist
                
                detailed_paths.append(detailed_path)
            
            return detailed_paths
            
        except nx.NetworkXNoPath:
            return []
    
    def save_graph(self, filepath: str):
        """Save the knowledge graph to a file"""
        try:
            # Save graph structure
            nx.write_gpickle(self.graph, filepath)
            
            # Save metadata separately
            metadata_file = filepath.replace('.gpickle', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.graph_metadata, f, indent=2)
            
            logger.info(f"Knowledge graph saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
    
    def load_graph(self, filepath: str):
        """Load the knowledge graph from a file"""
        try:
            # Load graph structure
            self.graph = nx.read_gpickle(filepath)
            
            # Load metadata
            metadata_file = filepath.replace('.gpickle', '_metadata.json')
            if Path(metadata_file).exists():
                with open(metadata_file, 'r') as f:
                    self.graph_metadata = json.load(f)
            
            logger.info(f"Knowledge graph loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        if not self.graph:
            return {}
        
        stats = {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'entity_types': self.graph_metadata.get('entity_types', {}),
            'relationship_types': self.graph_metadata.get('relationship_types', {}),
            'sources': len(self.graph_metadata.get('sources', [])),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'largest_component_size': len(max(nx.connected_components(self.graph.to_undirected()), key=len)) if self.graph.nodes else 0
        }
        
        return stats
