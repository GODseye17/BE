"""
Enhanced PubMed Filters Class
"""
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

class PubMedFilters:
    """Enhanced PubMed search filters with multi-topic boolean operator support"""
    
    def __init__(self):
        # Boolean operators supported by PubMed
        self.boolean_operators = {
            'AND': 'AND',
            'OR': 'OR', 
            'NOT': 'NOT'
        }
        
        # Publication date filters (PubMed standard)
        self.date_filters = {
            '1_year': '1 year',
            '5_years': '5 years', 
            '10_years': '10 years',
            'custom': 'custom'
        }
        
        # Text availability filters
        self.text_availability = {
            'abstract': 'hasabstract[text]',
            'full_text': 'full text[sb]',
            'free_full_text': 'free full text[sb]'
        }
        
        # Article types (PubMed publication types)
        self.article_types = {
            'clinical_trial': 'Clinical Trial[ptyp]',
            'randomized_controlled_trial': 'Randomized Controlled Trial[ptyp]',
            'meta_analysis': 'Meta-Analysis[ptyp]',
            'systematic_review': 'Systematic Review[ptyp]',
            'review': 'Review[ptyp]',
            'case_reports': 'Case Reports[ptyp]',
            'comparative_study': 'Comparative Study[ptyp]',
            'observational_study': 'Observational Study[ptyp]',
            'practice_guideline': 'Practice Guideline[ptyp]',
            'editorial': 'Editorial[ptyp]',
            'letter': 'Letter[ptyp]',
            'comment': 'Comment[ptyp]',
            'news': 'News[ptyp]',
            'biography': 'Biography[ptyp]',
            'congress': 'Congress[ptyp]',
            'consensus_development_conference': 'Consensus Development Conference[ptyp]',
            'guideline': 'Guideline[ptyp]'
        }
        
        # Language filters (PubMed supported languages)
        self.languages = {
            'english': 'english[lang]',
            'spanish': 'spanish[lang]',
            'french': 'french[lang]',
            'german': 'german[lang]',
            'italian': 'italian[lang]',
            'japanese': 'japanese[lang]',
            'portuguese': 'portuguese[lang]',
            'russian': 'russian[lang]',
            'chinese': 'chinese[lang]',
            'dutch': 'dutch[lang]',
            'polish': 'polish[lang]',
            'swedish': 'swedish[lang]',
            'danish': 'danish[lang]',
            'norwegian': 'norwegian[lang]',
            'finnish': 'finnish[lang]',
            'czech': 'czech[lang]',
            'hungarian': 'hungarian[lang]',
            'korean': 'korean[lang]',
            'turkish': 'turkish[lang]',
            'arabic': 'arabic[lang]',
            'hebrew': 'hebrew[lang]'
        }
        
        # Species filters
        self.species = {
            'humans': 'humans[mh]',
            'other_animals': 'animals[mh] NOT humans[mh]',
            'mice': 'mice[mh]',
            'rats': 'rats[mh]',
            'dogs': 'dogs[mh]',
            'cats': 'cats[mh]',
            'rabbits': 'rabbits[mh]',
            'primates': 'primates[mh]',
            'swine': 'swine[mh]',
            'sheep': 'sheep[mh]',
            'cattle': 'cattle[mh]'
        }
        
        # Sex filters
        self.sex = {
            'female': 'female[mh]',
            'male': 'male[mh]'
        }
        
        # Age filters (PubMed MeSH age groups)
        self.age_groups = {
            'child': 'child[mh]',  # birth-18 years
            'adult': 'adult[mh]',  # 19+ years  
            'aged': 'aged[mh]',    # 65+ years
            'infant': 'infant[mh]',  # birth-23 months
            'infant_newborn': 'infant, newborn[mh]',  # birth-1 month
            'child_preschool': 'child, preschool[mh]',  # 2-5 years
            'adolescent': 'adolescent[mh]',  # 13-18 years
            'young_adult': 'young adult[mh]',  # 19-24 years
            'middle_aged': 'middle aged[mh]',  # 45-64 years
            'aged_80_and_over': 'aged, 80 and over[mh]'  # 80+ years
        }
        
        # Other filters
        self.other_filters = {
            'exclude_preprints': 'NOT preprint[pt]',
            'medline': 'medline[sb]',
            'pubmed_not_medline': 'pubmed not medline[sb]',
            'in_process': 'in process[sb]',
            'publisher': 'publisher[sb]',
            'pmc': 'pmc[sb]',
            'nihms': 'nihms[sb]'
        }

    def sanitize_topic(self, topic: str) -> str:
        """Sanitize and validate individual topic for PubMed search"""
        if not topic or not topic.strip():
            return ""
        
        topic = topic.strip()
        
        # Handle quotes properly - if user wants phrase search, keep quotes
        if topic.startswith('"') and topic.endswith('"'):
            return topic  # Keep phrase search as is
        else:
            topic = topic.replace('"', '')
        
        # Clean up multiple spaces
        topic = re.sub(r'\s+', ' ', topic)
        return topic

    def build_multi_topic_query(self, topics: List[str], operator: str = 'AND', 
                               advanced_query: Optional[str] = None) -> str:
        """Build query from multiple topics with user-selected boolean operator"""
        if not topics and not advanced_query:
            raise ValueError("At least one topic or advanced query must be provided")
        
        # If advanced query is provided, use it (for power users)
        if advanced_query:
            return self.parse_boolean_query(advanced_query.strip())
        
        # Sanitize and filter out empty topics
        clean_topics = [self.sanitize_topic(topic) for topic in topics if topic and topic.strip()]
        
        if not clean_topics:
            raise ValueError("No valid topics provided")
        
        # Validate operator
        if operator not in self.boolean_operators:
            raise ValueError(f"Invalid operator '{operator}'. Must be one of: {list(self.boolean_operators.keys())}")
        
        # Handle special case for NOT operator
        if operator == 'NOT':
            if len(clean_topics) < 2:
                raise ValueError("NOT operator requires at least 2 topics")
            # For NOT, the first topic is included, rest are excluded
            base_topic = clean_topics[0]
            excluded_topics = clean_topics[1:]
            excluded_query = " AND ".join([f"NOT ({topic})" for topic in excluded_topics])
            return f"({base_topic}) AND ({excluded_query})"
        
        # For single topic, no operator needed
        if len(clean_topics) == 1:
            return f"({clean_topics[0]})"
        
        # Join multiple topics with selected operator
        boolean_op = self.boolean_operators[operator]
        joined_query = f" {boolean_op} ".join([f"({topic})" for topic in clean_topics])
        
        return f"({joined_query})"

    def build_date_filter(self, date_option: str, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> str:
        """Build date filter based on PubMed's date options"""
        if date_option == 'custom':
            if start_date and end_date:
                return f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
            elif start_date:
                return f'"{start_date}"[Date - Publication] : 3000[Date - Publication]'
            elif end_date:
                return f'1900[Date - Publication] : "{end_date}"[Date - Publication]'
        else:
            # Calculate date range for 1, 5, or 10 years
            years = int(date_option.split('_')[0])
            end_date = datetime.now().strftime('%Y/%m/%d')
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y/%m/%d')
            return f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        
        return ""

    def parse_boolean_query(self, query: str) -> str:
        """Parse and validate boolean operators in the query"""
        # Clean up the query - remove extra spaces
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Validate boolean operators are properly formatted
        query = re.sub(r'\s*\bAND\b\s*', ' AND ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s*\bOR\b\s*', ' OR ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s*\bNOT\b\s*', ' NOT ', query, flags=re.IGNORECASE)
        
        return query

    def build_complete_query(self, topics: Optional[List[str]] = None, operator: str = 'AND',
                           base_query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None,
                           advanced_query: Optional[str] = None) -> str:
        """Build complete PubMed query with multi-topic support and all filters"""
        # Determine base query method
        if advanced_query:
            processed_query = self.parse_boolean_query(advanced_query)
        elif topics:
            processed_query = self.build_multi_topic_query(topics, operator)
        elif base_query:
            processed_query = self.parse_boolean_query(base_query)
        else:
            raise ValueError("Must provide either topics, base_query, or advanced_query")
        
        if not filters:
            return processed_query
        
        filter_parts = []
        
        # Date filters
        if 'publication_date' in filters:
            date_filter = self.build_date_filter(
                filters['publication_date'],
                filters.get('custom_start_date'),
                filters.get('custom_end_date')
            )
            if date_filter:
                filter_parts.append(date_filter)
        
        # Text availability filters
        if 'text_availability' in filters:
            text_filters = []
            for availability in filters['text_availability']:
                if availability in self.text_availability:
                    text_filters.append(self.text_availability[availability])
            if text_filters:
                filter_parts.append(f'({" OR ".join(text_filters)})')
        
        # Article type filters
        if 'article_types' in filters:
            article_filters = []
            for article_type in filters['article_types']:
                if article_type in self.article_types:
                    article_filters.append(self.article_types[article_type])
            if article_filters:
                filter_parts.append(f'({" OR ".join(article_filters)})')
        
        # Language filters
        if 'languages' in filters:
            lang_filters = []
            for lang in filters['languages']:
                if lang in self.languages:
                    lang_filters.append(self.languages[lang])
            if lang_filters:
                filter_parts.append(f'({" OR ".join(lang_filters)})')
        
        # Species filters
        if 'species' in filters:
            species_filters = []
            for species in filters['species']:
                if species in self.species:
                    species_filters.append(self.species[species])
            if species_filters:
                filter_parts.append(f'({" OR ".join(species_filters)})')
        
        # Sex filters
        if 'sex' in filters:
            sex_filters = []
            for sex in filters['sex']:
                if sex in self.sex:
                    sex_filters.append(self.sex[sex])
            if sex_filters:
                filter_parts.append(f'({" OR ".join(sex_filters)})')
        
        # Age filters
        if 'age_groups' in filters:
            age_filters = []
            for age in filters['age_groups']:
                if age in self.age_groups:
                    age_filters.append(self.age_groups[age])
            if age_filters:
                filter_parts.append(f'({" OR ".join(age_filters)})')
        
        # Other filters
        if 'other_filters' in filters:
            for other_filter in filters['other_filters']:
                if other_filter in self.other_filters:
                    filter_parts.append(self.other_filters[other_filter])
        
        # Custom filters with boolean operators
        if 'custom_filters' in filters:
            for custom_filter in filters['custom_filters']:
                processed_custom = self.parse_boolean_query(custom_filter)
                filter_parts.append(f'({processed_custom})')
        
        # Combine base query with filters
        if filter_parts:
            final_query = f'{processed_query} AND ({" AND ".join(filter_parts)})'
        else:
            final_query = processed_query
        
        return final_query