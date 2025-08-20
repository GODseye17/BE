"""
Medical Relationship Extraction for Knowledge Graph Construction
"""
import logging
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RelationshipExtractor:
    """Extract medical relationships from text for knowledge graph construction"""
    
    def __init__(self):
        # Medical relationship patterns
        self.relationship_patterns = {
            'treats': [
                r'\b(?:treats?|treatment for|therapy for|medication for|drug for|cure for|remedy for)\b',
                r'\b(?:effective against|successful in treating|used to treat|prescribed for|indicated for)\b'
            ],
            'causes': [
                r'\b(?:causes?|leads to|results in|brings about|triggers?|induces?|provokes?)\b',
                r'\b(?:risk factor for|associated with|linked to|correlated with|related to)\b'
            ],
            'inhibits': [
                r'\b(?:inhibits?|blocks?|prevents?|suppresses?|reduces?|decreases?|lowers?)\b',
                r'\b(?:antagonist|inhibitor|blocker|suppressor|reducer)\b'
            ],
            'activates': [
                r'\b(?:activates?|stimulates?|enhances?|increases?|promotes?|boosts?|upregulates?)\b',
                r'\b(?:agonist|activator|stimulator|enhancer|promoter)\b'
            ],
            'diagnoses': [
                r'\b(?:diagnoses?|detects?|identifies?|screens? for|tests? for|examines? for)\b',
                r'\b(?:diagnostic|detection|identification|screening|testing)\b'
            ],
            'affects': [
                r'\b(?:affects?|impacts?|influences?|modifies?|alters?|changes?|affects?)\b',
                r'\b(?:effect on|impact on|influence on|modification of|alteration of)\b'
            ],
            'interacts_with': [
                r'\b(?:interacts? with|combines? with|works? with|cooperates? with|synergizes? with)\b',
                r'\b(?:interaction|combination|cooperation|synergy|synergistic)\b'
            ],
            'contraindicated_for': [
                r'\b(?:contraindicated for|not recommended for|avoided in|not suitable for|unsafe for)\b',
                r'\b(?:contraindication|avoid|unsafe|not recommended|not suitable)\b'
            ]
        }
        
        # Medical action verbs
        self.action_verbs = {
            'treats': ['treat', 'cure', 'heal', 'manage', 'control'],
            'causes': ['cause', 'lead', 'result', 'trigger', 'induce'],
            'inhibits': ['inhibit', 'block', 'prevent', 'suppress', 'reduce'],
            'activates': ['activate', 'stimulate', 'enhance', 'increase', 'promote'],
            'diagnoses': ['diagnose', 'detect', 'identify', 'screen', 'test'],
            'affects': ['affect', 'impact', 'influence', 'modify', 'alter'],
            'interacts_with': ['interact', 'combine', 'work', 'cooperate', 'synergize'],
            'contraindicated_for': ['contraindicate', 'avoid', 'unsafe', 'not recommend']
        }
    
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities from text"""
        if not text or not entities:
            return []
        
        relationships = []
        
        # Extract relationships using pattern matching
        for relationship_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Find entities around the relationship
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    
                    # Find entities in the context
                    context_entities = self._find_entities_in_context(context, entities)
                    
                    if len(context_entities) >= 2:
                        # Create relationship
                        relationship = {
                            'source': context_entities[0],
                            'target': context_entities[1],
                            'relationship': relationship_type,
                            'confidence': self._calculate_confidence(match, context),
                            'context': context.strip(),
                            'source_type': self._get_entity_type(context_entities[0], entities),
                            'target_type': self._get_entity_type(context_entities[1], entities)
                        }
                        relationships.append(relationship)
        
        # Extract relationships using dependency parsing (simplified)
        dependency_relationships = self._extract_dependency_relationships(text, entities)
        relationships.extend(dependency_relationships)
        
        # Remove duplicates and filter by confidence
        unique_relationships = self._deduplicate_relationships(relationships)
        filtered_relationships = [r for r in unique_relationships if r['confidence'] > 0.3]
        
        logger.debug(f"Extracted {len(filtered_relationships)} relationships")
        return filtered_relationships
    
    def _find_entities_in_context(self, context: str, entities: Dict[str, List[str]]) -> List[str]:
        """Find entities mentioned in the given context"""
        found_entities = []
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if re.search(r'\b' + re.escape(entity) + r'\b', context, re.IGNORECASE):
                    found_entities.append(entity)
        
        # Sort by position in text (earlier entities first)
        found_entities.sort(key=lambda x: context.lower().find(x.lower()))
        return found_entities
    
    def _calculate_confidence(self, match, context: str) -> float:
        """Calculate confidence score for a relationship"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for exact matches
        if match.group().lower() in ['treats', 'causes', 'inhibits', 'activates']:
            confidence += 0.3
        
        # Higher confidence for medical terminology
        medical_terms = ['therapy', 'treatment', 'medication', 'drug', 'diagnosis', 'screening']
        if any(term in context.lower() for term in medical_terms):
            confidence += 0.2
        
        # Lower confidence for negated statements
        if re.search(r'\b(?:not|no|never|doesn\'t|don\'t|isn\'t|aren\'t)\b', context, re.IGNORECASE):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _get_entity_type(self, entity: str, entities: Dict[str, List[str]]) -> str:
        """Get the type of an entity"""
        for entity_type, entity_list in entities.items():
            if entity in entity_list:
                return entity_type
        return 'UNKNOWN'
    
    def _extract_dependency_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract relationships using simplified dependency parsing"""
        relationships = []
        
        # Simple pattern-based dependency extraction
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Find entities in sentence
            sentence_entities = self._find_entities_in_context(sentence, entities)
            
            if len(sentence_entities) >= 2:
                # Look for action verbs between entities
                for action_type, verbs in self.action_verbs.items():
                    for verb in verbs:
                        if re.search(r'\b' + verb + r'\b', sentence, re.IGNORECASE):
                            relationship = {
                                'source': sentence_entities[0],
                                'target': sentence_entities[1],
                                'relationship': action_type,
                                'confidence': 0.4,  # Lower confidence for dependency-based
                                'context': sentence,
                                'source_type': self._get_entity_type(sentence_entities[0], entities),
                                'target_type': self._get_entity_type(sentence_entities[1], entities)
                            }
                            relationships.append(relationship)
                            break
        
        return relationships
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a unique key for each relationship
            key = (rel['source'], rel['target'], rel['relationship'])
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
            else:
                # If duplicate found, keep the one with higher confidence
                existing_rel = next(r for r in unique_relationships if (r['source'], r['target'], r['relationship']) == key)
                if rel['confidence'] > existing_rel['confidence']:
                    unique_relationships.remove(existing_rel)
                    unique_relationships.append(rel)
        
        return unique_relationships
    
    def validate_relationship(self, relationship: Dict[str, Any]) -> bool:
        """Validate if a relationship makes medical sense"""
        source_type = relationship.get('source_type', '')
        target_type = relationship.get('target_type', '')
        rel_type = relationship.get('relationship', '')
        
        # Medical relationship validation rules
        valid_combinations = {
            'treats': [
                ('DRUG', 'DISEASE'),
                ('PROCEDURE', 'DISEASE'),
                ('DRUG', 'SYMPTOM')
            ],
            'causes': [
                ('DISEASE', 'SYMPTOM'),
                ('GENE', 'DISEASE'),
                ('DRUG', 'SYMPTOM')  # Side effects
            ],
            'inhibits': [
                ('DRUG', 'GENE'),
                ('DRUG', 'PROTEIN'),
                ('GENE', 'GENE')
            ],
            'activates': [
                ('DRUG', 'GENE'),
                ('GENE', 'GENE'),
                ('PROCEDURE', 'GENE')
            ],
            'diagnoses': [
                ('PROCEDURE', 'DISEASE'),
                ('PROCEDURE', 'SYMPTOM')
            ],
            'affects': [
                ('DISEASE', 'ORGAN'),
                ('DRUG', 'ORGAN'),
                ('PROCEDURE', 'ORGAN')
            ],
            'interacts_with': [
                ('DRUG', 'DRUG'),
                ('GENE', 'GENE'),
                ('DRUG', 'GENE')
            ],
            'contraindicated_for': [
                ('DRUG', 'DISEASE'),
                ('PROCEDURE', 'DISEASE')
            ]
        }
        
        if rel_type in valid_combinations:
            valid_pairs = valid_combinations[rel_type]
            return (source_type, target_type) in valid_pairs
        
        return True  # Allow unknown relationship types
