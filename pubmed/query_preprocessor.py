"""
Query Preprocessor for transforming natural language to PubMed queries
"""
import re
import logging
from typing import Dict, Optional, Tuple
from core.globals import get_globals

logger = logging.getLogger(__name__)

class QueryPreprocessor:
    """Transform natural language research questions into optimized PubMed queries"""
    
    def __init__(self):
        # Common medical synonyms for expansion
        self.medical_synonyms = {
            "heart attack": "myocardial infarction",
            "high blood pressure": "hypertension",
            "low blood pressure": "hypotension",
            "sugar": "glucose",
            "diabetes": "diabetes mellitus",
            "cancer": "neoplasm",
            "tumor": "neoplasm",
            "stroke": "cerebrovascular accident",
            "copd": "chronic obstructive pulmonary disease",
            "ptsd": "post traumatic stress disorder",
            "adhd": "attention deficit hyperactivity disorder",
            "mrsa": "methicillin resistant staphylococcus aureus",
            "uti": "urinary tract infection",
            "mi": "myocardial infarction",
            "hiv": "human immunodeficiency virus",
            "aids": "acquired immunodeficiency syndrome",
            "covid": "COVID-19",
            "covid-19": "SARS-CoV-2",
            "coronavirus": "COVID-19",
            "covid vaccine": "COVID-19 vaccines",
            "covid-19 vaccine": "SARS-CoV-2 vaccine",
            "covid vaccines": "COVID-19 vaccines"
        }
        
        # Query patterns and their templates
        self.query_templates = {
            "effect_of_X_on_Y": {
                "patterns": [
                    r"(?:what is the |what's the |)effect(?:s|) of (.+?) on (.+?)(?:\?|$)",
                    r"how does (.+?) affect (.+?)(?:\?|$)",
                    r"impact of (.+?) on (.+?)(?:\?|$)"
                ],
                "template": "({X}[MeSH] OR {X}[Title/Abstract]) AND ({Y}[MeSH] OR {Y}[Title/Abstract])"
            },
            "X_vs_Y": {
                "patterns": [
                    r"(.+?) (?:vs\.?|versus|compared to|compared with) (.+?)(?:\?|$)",
                    r"difference(?:s|) between (.+?) and (.+?)(?:\?|$)",
                    r"comparison of (.+?) and (.+?)(?:\?|$)"
                ],
                "template": "({X}[Title/Abstract] OR {Y}[Title/Abstract]) AND (comparison[MeSH] OR versus[Title/Abstract] OR compared[Title/Abstract])"
            },
            "treatment_for_X": {
                "patterns": [
                    r"(?:treatment|therapy|therapies|medication|medicine) (?:for|of) (.+?)(?:\?|$)",
                    r"how to treat (.+?)(?:\?|$)",
                    r"(.+?) treatment options",
                    r"what treats (.+?)(?:\?|$)"
                ],
                "template": "({X}[MeSH] OR {X}[Title/Abstract]) AND (therapeutics[MeSH] OR therapy[MeSH] OR treatment[Title/Abstract])"
            },
            "side_effects_of_X": {
                "patterns": [
                    r"(.+?) side effect(?:s|)(?:\?|$)",
                    r"side effect(?:s|) of (.+?)(?:\?|$)",
                    r"adverse effect(?:s|) of (.+?)(?:\?|$)",
                    r"(.+?) adverse effect(?:s|)(?:\?|$)",
                    r"risks of (.+?)(?:\?|$)"
                ],
                "template": "({X}[MeSH] OR {X}[Title/Abstract]) AND (adverse effects[MeSH] OR side effects[Title/Abstract] OR adverse events[Title/Abstract])"
            },
            "mechanism_of_X": {
                "patterns": [
                    r"how does (.+?) work(?:\?|$)",
                    r"mechanism of (.+?)(?:\?|$)",
                    r"(.+?) mechanism of action",
                    r"pathophysiology of (.+?)(?:\?|$)"
                ],
                "template": "({X}[MeSH] OR {X}[Title/Abstract]) AND (mechanism[MeSH] OR mechanism of action[Title/Abstract] OR pathophysiology[MeSH])"
            },
            "diagnosis_of_X": {
                "patterns": [
                    r"how to diagnose (.+?)(?:\?|$)",
                    r"diagnosis of (.+?)(?:\?|$)",
                    r"(.+?) diagnosis",
                    r"diagnostic (?:test|tests|criteria) for (.+?)(?:\?|$)"
                ],
                "template": "({X}[MeSH] OR {X}[Title/Abstract]) AND (diagnosis[MeSH] OR diagnostic[Title/Abstract])"
            },
            "X_in_Y_population": {
                "patterns": [
                    r"(.+?) in (.+?)(?:\?|$)",
                    r"(.+?) among (.+?)(?:\?|$)"
                ],
                "template": "({X}[MeSH] OR {X}[Title/Abstract]) AND ({Y}[MeSH] OR {Y}[Title/Abstract])"
            }
        }
        
        # Keywords that indicate natural language query
        self.natural_language_indicators = [
            '?', ' is ', ' are ', ' was ', ' were ', 'how ', 'what ', 'when ', 'where ', 
            'why ', 'which ', 'effect', 'relationship', 'between', 'among', 'cause',
            'treatment', 'diagnos', 'prevent', 'risk', "what's", "how's", "i'm",
            "i am", "research", "studying", "looking for", "interested in"
        ]
        
    def looks_like_natural_language(self, text: str) -> bool:
        """Detect if the query is natural language or already PubMed syntax"""
        text_lower = text.lower()
        
        # If it has PubMed syntax elements, it's not natural language
        pubmed_syntax = ['[mesh]', '[title/abstract]', '[ptyp]', ' and ', ' or ', ' not ']
        if any(syntax in text_lower for syntax in pubmed_syntax):
            return False
            
        # Check for natural language indicators
        return any(indicator in text_lower for indicator in self.natural_language_indicators)
    
    def expand_with_synonyms(self, term: str) -> str:
        """Expand a term with medical synonyms"""
        term_lower = term.lower().strip()
        
        # Direct synonym lookup
        if term_lower in self.medical_synonyms:
            synonym = self.medical_synonyms[term_lower]
            return f"({term} OR {synonym})"
        
        # Check if term contains any known terms
        for original, synonym in self.medical_synonyms.items():
            if original in term_lower:
                expanded = term_lower.replace(original, f"{original} OR {synonym}")
                return f"({expanded})"
        
        return term
    
    def extract_pattern_match(self, query: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """Extract pattern matches from query"""
        query_lower = query.lower().strip()
        
        # Temporal words to clean from medical terms
        temporal_words = ['latest', 'recent', 'new', 'current', 'emerging', 'novel']
        
        for pattern_name, pattern_info in self.query_templates.items():
            for pattern in pattern_info['patterns']:
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    # Clean temporal words from extracted terms
                    cleaned_groups = []
                    for group in groups:
                        cleaned_term = group.strip()
                        # Remove temporal words from the beginning
                        for temporal in temporal_words:
                            if cleaned_term.startswith(f"{temporal} "):
                                cleaned_term = cleaned_term[len(temporal)+1:]
                        cleaned_groups.append(cleaned_term)
                    
                    if pattern_name == "X_in_Y_population":
                        # Special handling for population patterns
                        if any(pop in cleaned_groups[1] for pop in ['children', 'elderly', 'women', 'men', 'adults', 'patients']):
                            return pattern_name, {"X": cleaned_groups[0], "Y": cleaned_groups[1]}
                    elif len(cleaned_groups) == 1:
                        return pattern_name, {"X": cleaned_groups[0]}
                    elif len(cleaned_groups) == 2:
                        return pattern_name, {"X": cleaned_groups[0], "Y": cleaned_groups[1]}
        
        return None
    
    def apply_template(self, template_name: str, variables: Dict[str, str]) -> str:
        """Apply a query template with variables"""
        template = self.query_templates[template_name]['template']
        
        # Expand terms with synonyms
        for var_name, var_value in variables.items():
            expanded_value = self.expand_with_synonyms(var_value)
            template = template.replace(f"{{{var_name}}}", expanded_value)
        
        return template
    
    def add_temporal_filters(self, query: str, original_text: str) -> str:
        """Add temporal filters based on keywords"""
        text_lower = original_text.lower()
        
        if any(word in text_lower for word in ['latest', 'recent', 'new', 'current', 'emerging']):
            # Add 5-year filter for recent research
            return f"{query} AND (\"2019\"[Date - Publication] : \"3000\"[Date - Publication])"
        elif 'last year' in text_lower:
            return f"{query} AND (\"2023\"[Date - Publication] : \"3000\"[Date - Publication])"
        elif 'last 5 years' in text_lower or 'past 5 years' in text_lower:
            return f"{query} AND (\"2019\"[Date - Publication] : \"3000\"[Date - Publication])"
        elif 'last 10 years' in text_lower or 'past 10 years' in text_lower:
            return f"{query} AND (\"2014\"[Date - Publication] : \"3000\"[Date - Publication])"
        
        return query
    
    def add_study_type_filters(self, query: str, original_text: str) -> str:
        """Add study type filters based on keywords"""
        text_lower = original_text.lower()
        
        study_types = {
            'clinical trial': 'Clinical Trial[ptyp]',
            'randomized': 'Randomized Controlled Trial[ptyp]',
            'meta-analysis': 'Meta-Analysis[ptyp]',
            'systematic review': 'Systematic Review[ptyp]',
            'review': 'Review[ptyp]',
            'case study': 'Case Reports[ptyp]',
            'cohort': 'Cohort Studies[ptyp]'
        }
        
        for keyword, filter_text in study_types.items():
            if keyword in text_lower:
                return f"{query} AND {filter_text}"
        
        return query
    
    def transform_with_llm(self, user_query: str) -> str:
        """Use LLM to transform natural language to PubMed query"""
        try:
            globals_dict = get_globals()
            llm = globals_dict.get('llm')
            
            if not llm:
                logger.warning("LLM not available for query transformation")
                return user_query
            
            prompt = """You are a medical search expert. Convert this natural language research question into an effective PubMed search query.

Rules:
1. Use MeSH terms with [MeSH] when possible
2. Use [Title/Abstract] for non-MeSH terms
3. Use AND to connect different concepts
4. Use OR to connect synonyms
5. Add relevant medical synonyms
6. Keep it comprehensive but focused

User query: {query}

Return ONLY the PubMed query, no explanation or additional text."""

            # Use the LLM to generate the query
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt.format(query=user_query))]
            result = llm._generate(messages)
            
            if result and result.generations and result.generations[0].message.content:
                transformed = result.generations[0].message.content.strip()
                logger.info(f"LLM transformed query: '{user_query}' -> '{transformed}'")
                return transformed
            
        except Exception as e:
            logger.error(f"Error in LLM transformation: {e}")
        
        return user_query
    
    def transform_natural_to_pubmed(self, user_query: str) -> str:
        """Main method to transform natural language to PubMed query"""
        # Skip if already PubMed syntax
        if not self.looks_like_natural_language(user_query):
            logger.info(f"Query already in PubMed syntax: {user_query}")
            return user_query
        
        logger.info(f"Transforming natural language query: {user_query}")
        
        # First, try pattern matching
        pattern_match = self.extract_pattern_match(user_query)
        if pattern_match:
            template_name, variables = pattern_match
            transformed_query = self.apply_template(template_name, variables)
            logger.info(f"Pattern-based transformation: '{user_query}' -> '{transformed_query}'")
        else:
            # If no pattern matches, use LLM
            transformed_query = self.transform_with_llm(user_query)
        
        # Add temporal filters
        transformed_query = self.add_temporal_filters(transformed_query, user_query)
        
        # Add study type filters
        transformed_query = self.add_study_type_filters(transformed_query, user_query)
        
        logger.info(f"Final transformed query: {transformed_query}")
        return transformed_query
    
    def get_query_explanation(self, original: str, transformed: str) -> str:
        """Generate explanation for the transformation"""
        explanation = f"Original: {original}\n"
        explanation += f"Transformed: {transformed}\n\n"
        
        if "[MeSH]" in transformed:
            explanation += "• Added MeSH terms for precise medical concept matching\n"
        if " OR " in transformed:
            explanation += "• Included synonyms to capture more relevant articles\n"
        if "[Date - Publication]" in transformed:
            explanation += "• Added date filter for recent research\n"
        if "[ptyp]" in transformed:
            explanation += "• Filtered by study type for higher quality evidence\n"
        
        return explanation