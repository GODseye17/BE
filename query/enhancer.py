"""
Query Enhancement for PubMed Searches
"""
import logging
import re
from typing import List, Dict, Any, Optional

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
