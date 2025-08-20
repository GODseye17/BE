"""
Medical Entity Extraction for Knowledge Graph Construction
"""
import logging
import re
from typing import List, Dict, Any, Set
from sentence_transformers import SentenceTransformer
import spacy

logger = logging.getLogger(__name__)

class MedicalEntityExtractor:
    """Extract medical entities from text for knowledge graph construction"""
    
    def __init__(self):
        # Initialize medical NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for entity extraction")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using rule-based extraction")
            self.nlp = None
        
        # Medical entity patterns
        self.entity_patterns = {
            'DISEASE': [
                r'\b(?:diabetes|hypertension|cancer|stroke|heart attack|asthma|arthritis|depression|anxiety|obesity|diabetes mellitus|type 1 diabetes|type 2 diabetes|cardiovascular disease|coronary artery disease|chronic kidney disease|chronic obstructive pulmonary disease)\b',
                r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:disease|syndrome|disorder|condition)\b'
            ],
            'DRUG': [
                r'\b(?:metformin|insulin|aspirin|ibuprofen|acetaminophen|morphine|penicillin|amoxicillin|atorvastatin|lisinopril|amlodipine|omeprazole|prednisone|warfarin|heparin)\b',
                r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?:\s+(?:tablet|capsule|injection|cream|ointment))?\b'
            ],
            'GENE': [
                r'\b(?:BRCA1|BRCA2|TP53|APC|KRAS|EGFR|HER2|BRAF|ALK|ROS1|PDL1|MSH2|MLH1|BRCA|TP53|APC|KRAS|EGFR|HER2|BRAF|ALK|ROS1|PDL1|MSH2|MLH1)\b',
                r'\b[A-Z]{2,6}\d*\b'  # Gene symbols like ABC1, DEF2
            ],
            'SYMPTOM': [
                r'\b(?:pain|fatigue|nausea|vomiting|fever|headache|dizziness|shortness of breath|cough|chest pain|abdominal pain|back pain|joint pain|muscle pain|swelling|inflammation|bleeding|bruising|rash|itching|numbness|tingling|weakness|paralysis|seizure|confusion|memory loss|depression|anxiety|insomnia|weight loss|weight gain|loss of appetite|increased appetite|thirst|frequent urination|constipation|diarrhea)\b'
            ],
            'PROCEDURE': [
                r'\b(?:surgery|operation|procedure|biopsy|endoscopy|colonoscopy|mammography|ultrasound|MRI|CT scan|X-ray|blood test|urine test|stool test|vaccination|immunization|chemotherapy|radiotherapy|radiation therapy|transplant|transplantation|dialysis|dialysis treatment|cardiac catheterization|angioplasty|stent placement|pacemaker|defibrillator|bypass surgery|heart transplant|kidney transplant|liver transplant|bone marrow transplant)\b'
            ],
            'ORGAN': [
                r'\b(?:heart|liver|kidney|lung|brain|stomach|intestine|colon|pancreas|spleen|gallbladder|bladder|prostate|uterus|ovary|testicle|breast|thyroid|adrenal|pituitary|hypothalamus|bone|muscle|skin|eye|ear|nose|throat|esophagus|trachea|bronchus|artery|vein|lymph node|bone marrow|spinal cord|nerve)\b'
            ]
        }
        
        # Medical acronyms and their full forms
        self.medical_acronyms = {
            'MI': 'myocardial infarction',
            'CVD': 'cardiovascular disease',
            'CAD': 'coronary artery disease',
            'CHF': 'congestive heart failure',
            'COPD': 'chronic obstructive pulmonary disease',
            'CKD': 'chronic kidney disease',
            'ESRD': 'end stage renal disease',
            'IBD': 'inflammatory bowel disease',
            'UC': 'ulcerative colitis',
            'CD': 'crohn disease',
            'RA': 'rheumatoid arthritis',
            'OA': 'osteoarthritis',
            'SLE': 'systemic lupus erythematosus',
            'MS': 'multiple sclerosis',
            'PD': 'parkinson disease',
            'AD': 'alzheimer disease',
            'PTSD': 'post traumatic stress disorder',
            'OCD': 'obsessive compulsive disorder',
            'ADHD': 'attention deficit hyperactivity disorder',
            'ASD': 'autism spectrum disorder'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        if not text:
            return {}
        
        entities = {entity_type: [] for entity_type in self.entity_patterns.keys()}
        
        # Extract entities using patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = match.group().strip()
                    if entity and entity not in entities[entity_type]:
                        entities[entity_type].append(entity)
        
        # Expand acronyms
        expanded_entities = self._expand_acronyms(entities, text)
        
        # Use spaCy for additional entity extraction if available
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(text)
            for entity_type, entity_list in spacy_entities.items():
                if entity_type in expanded_entities:
                    expanded_entities[entity_type].extend(entity_list)
                else:
                    expanded_entities[entity_type] = entity_list
        
        # Remove duplicates and clean up
        cleaned_entities = {}
        for entity_type, entity_list in expanded_entities.items():
            unique_entities = list(set([entity.strip() for entity in entity_list if entity.strip()]))
            if unique_entities:
                cleaned_entities[entity_type] = unique_entities
        
        logger.debug(f"Extracted entities: {cleaned_entities}")
        return cleaned_entities
    
    def _expand_acronyms(self, entities: Dict[str, List[str]], text: str) -> Dict[str, List[str]]:
        """Expand medical acronyms to their full forms"""
        expanded_entities = entities.copy()
        
        for acronym, full_form in self.medical_acronyms.items():
            if re.search(r'\b' + acronym + r'\b', text, re.IGNORECASE):
                # Add full form to appropriate entity type
                if 'DISEASE' in expanded_entities:
                    expanded_entities['DISEASE'].append(full_form)
                elif 'SYMPTOM' in expanded_entities:
                    expanded_entities['SYMPTOM'].append(full_form)
        
        return expanded_entities
    
    def _extract_spacy_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        spacy_entities = {}
        
        for ent in doc.ents:
            # Map spaCy entity types to our medical types
            if ent.label_ in ['DISEASE', 'CONDITION']:
                entity_type = 'DISEASE'
            elif ent.label_ in ['DRUG', 'MEDICATION']:
                entity_type = 'DRUG'
            elif ent.label_ in ['GENE', 'PROTEIN']:
                entity_type = 'GENE'
            elif ent.label_ in ['SYMPTOM', 'SIGN']:
                entity_type = 'SYMPTOM'
            elif ent.label_ in ['PROCEDURE', 'TREATMENT']:
                entity_type = 'PROCEDURE'
            elif ent.label_ in ['ORGAN', 'BODY_PART']:
                entity_type = 'ORGAN'
            else:
                continue
            
            if entity_type not in spacy_entities:
                spacy_entities[entity_type] = []
            
            entity_text = ent.text.strip()
            if entity_text and entity_text not in spacy_entities[entity_type]:
                spacy_entities[entity_type].append(entity_text)
        
        return spacy_entities
    
    def get_entity_relationships(self, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate potential relationships between extracted entities"""
        relationships = []
        
        # Disease-Symptom relationships
        if 'DISEASE' in entities and 'SYMPTOM' in entities:
            for disease in entities['DISEASE']:
                for symptom in entities['SYMPTOM']:
                    relationships.append({
                        'source': disease,
                        'target': symptom,
                        'relationship': 'causes',
                        'source_type': 'DISEASE',
                        'target_type': 'SYMPTOM'
                    })
        
        # Drug-Disease relationships
        if 'DRUG' in entities and 'DISEASE' in entities:
            for drug in entities['DRUG']:
                for disease in entities['DISEASE']:
                    relationships.append({
                        'source': drug,
                        'target': disease,
                        'relationship': 'treats',
                        'source_type': 'DRUG',
                        'target_type': 'DISEASE'
                    })
        
        # Gene-Disease relationships
        if 'GENE' in entities and 'DISEASE' in entities:
            for gene in entities['GENE']:
                for disease in entities['DISEASE']:
                    relationships.append({
                        'source': gene,
                        'target': disease,
                        'relationship': 'associated_with',
                        'source_type': 'GENE',
                        'target_type': 'DISEASE'
                    })
        
        # Organ-Disease relationships
        if 'ORGAN' in entities and 'DISEASE' in entities:
            for organ in entities['ORGAN']:
                for disease in entities['DISEASE']:
                    relationships.append({
                        'source': disease,
                        'target': organ,
                        'relationship': 'affects',
                        'source_type': 'DISEASE',
                        'target_type': 'ORGAN'
                    })
        
        return relationships
