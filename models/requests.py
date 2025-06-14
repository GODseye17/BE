"""
Request models for the Vivum RAG Backend
"""
from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from .enums import (
    BooleanOperator, PublicationDate, TextAvailability, ArticleType,
    Language, Species, Sex, AgeGroup, OtherFilter, SortBy, SearchField
)

class PubMedFiltersModel(BaseModel):
    publication_date: Optional[PublicationDate] = None
    custom_start_date: Optional[str] = Field(None, pattern=r'^\d{4}/\d{2}/\d{2}$', description="Date in YYYY/MM/DD format")
    custom_end_date: Optional[str] = Field(None, pattern=r'^\d{4}/\d{2}/\d{2}$', description="Date in YYYY/MM/DD format")
    text_availability: Optional[List[TextAvailability]] = None
    article_types: Optional[List[ArticleType]] = None
    languages: Optional[List[Language]] = None
    species: Optional[List[Species]] = None
    sex: Optional[List[Sex]] = None
    age_groups: Optional[List[AgeGroup]] = None
    other_filters: Optional[List[OtherFilter]] = None
    custom_filters: Optional[List[str]] = None
    sort_by: Optional[SortBy] = SortBy.RELEVANCE
    search_field: Optional[SearchField] = SearchField.TITLE_ABSTRACT
    
    # FIXED: Handle empty strings properly
    @field_validator('publication_date', mode='before')
    @classmethod
    def validate_publication_date(cls, v):
        if v == "" or v is None:
            return None
        return v
    
    @field_validator('custom_start_date', 'custom_end_date', mode='before')
    @classmethod
    def validate_date_strings(cls, v):
        if v == "" or v is None:
            return None
        # Validate date format if not empty
        if v is not None:
            try:
                datetime.strptime(v, '%Y/%m/%d')
            except ValueError:
                raise ValueError('Date must be in YYYY/MM/DD format')
        return v
    
    @field_validator('text_availability', 'article_types', 'languages', 'species', 'sex', 'age_groups', 'other_filters', 'custom_filters', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        if v == "" or v == [] or v is None:
            return None
        return v
    
    @field_validator('sort_by', 'search_field', mode='before')
    @classmethod
    def validate_enum_fields(cls, v):
        if v == "" or v is None:
            return None
        return v

class TopicRequest(BaseModel):
    # Multi-topic search fields (NEW)
    topics: Optional[List[str]] = Field(None, description="List of search topics for multi-topic search")
    operator: Optional[BooleanOperator] = Field(BooleanOperator.AND, description="Boolean operator to combine topics")
    
    # Single topic search field (BACKWARD COMPATIBILITY)
    topic: Optional[str] = Field(None, description="Single search topic (backward compatibility)")
    
    # Advanced query field (NEW)
    advanced_query: Optional[str] = Field(None, description="Advanced PubMed query string")
    
    # Other fields
    source: Optional[str] = Field("pubmed", description="Data source (pubmed/scopus)")
    max_results: Optional[int] = Field(20, ge=1, le=10000, description="Maximum number of results")
    filters: Optional[PubMedFiltersModel] = None

    @model_validator(mode='before')
    @classmethod
    def validate_search_input(cls, data):
        if isinstance(data, dict):
            values = data
        else:
            return data
            
        topics = values.get('topics')
        topic = values.get('topic')
        advanced_query = values.get('advanced_query')
        operator = values.get('operator')
        
        # Count non-empty search inputs
        search_inputs = [
            bool(topics and any(t.strip() for t in topics if t)),
            bool(topic and topic.strip()),
            bool(advanced_query and advanced_query.strip())
        ]
        
        if sum(search_inputs) == 0:
            raise ValueError('Must provide either topics, topic, or advanced_query')
        
        if sum(search_inputs) > 1:
            raise ValueError('Can only use one search method: topics, topic, or advanced_query')
        
        # Validate NOT operator requirements
        if operator == "NOT" and topics and len(topics) < 2:
            raise ValueError('NOT operator requires at least 2 topics')
        
        # Validate topics are not empty
        if topics:
            clean_topics = [t.strip() for t in topics if t and t.strip()]
            if not clean_topics:
                raise ValueError('Topics cannot be empty')
            values['topics'] = clean_topics
        
        return values

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "topics": ["diabetes", "insulin therapy", "type 2"],
                    "operator": "AND",
                    "max_results": 50,
                    "filters": {
                        "publication_date": "5_years",
                        "article_types": ["randomized_controlled_trial", "meta_analysis"],
                        "languages": ["english"],
                        "species": ["humans"]
                    }
                },
                {
                    "topic": "machine learning in healthcare",
                    "max_results": 30,
                    "filters": {
                        "publication_date": "2_years",
                        "languages": ["english"]
                    }
                },
            ]
        }
    }

class QueryRequest(BaseModel):
    query: str
    topic_id: str
    conversation_id: Optional[str] = None