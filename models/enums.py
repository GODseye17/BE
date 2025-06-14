"""
Enums for the Vivum RAG Backend
"""
from enum import Enum

class BooleanOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

class PublicationDate(str, Enum):
    ONE_YEAR = "1_year"
    FIVE_YEARS = "5_years"
    TEN_YEARS = "10_years"
    CUSTOM = "custom"

class TextAvailability(str, Enum):
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    FREE_FULL_TEXT = "free_full_text"

class ArticleType(str, Enum):
    CLINICAL_TRIAL = "clinical_trial"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    REVIEW = "review"
    CASE_REPORTS = "case_reports"
    COMPARATIVE_STUDY = "comparative_study"
    OBSERVATIONAL_STUDY = "observational_study"
    PRACTICE_GUIDELINE = "practice_guideline"
    EDITORIAL = "editorial"
    LETTER = "letter"
    COMMENT = "comment"
    NEWS = "news"
    BIOGRAPHY = "biography"
    CONGRESS = "congress"
    CONSENSUS_DEVELOPMENT_CONFERENCE = "consensus_development_conference"
    GUIDELINE = "guideline"

class Language(str, Enum):
    ENGLISH = "english"
    SPANISH = "spanish"
    FRENCH = "french"
    GERMAN = "german"
    ITALIAN = "italian"
    JAPANESE = "japanese"
    PORTUGUESE = "portuguese"
    RUSSIAN = "russian"
    CHINESE = "chinese"
    DUTCH = "dutch"
    POLISH = "polish"
    SWEDISH = "swedish"
    DANISH = "danish"
    NORWEGIAN = "norwegian"
    FINNISH = "finnish"
    CZECH = "czech"
    HUNGARIAN = "hungarian"
    KOREAN = "korean"
    TURKISH = "turkish"
    ARABIC = "arabic"
    HEBREW = "hebrew"

class Species(str, Enum):
    HUMANS = "humans"
    OTHER_ANIMALS = "other_animals"
    MICE = "mice"
    RATS = "rats"
    DOGS = "dogs"
    CATS = "cats"
    RABBITS = "rabbits"
    PRIMATES = "primates"
    SWINE = "swine"
    SHEEP = "sheep"
    CATTLE = "cattle"

class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"

class AgeGroup(str, Enum):
    CHILD = "child"
    ADULT = "adult"
    AGED = "aged"
    INFANT = "infant"
    INFANT_NEWBORN = "infant_newborn"
    CHILD_PRESCHOOL = "child_preschool"
    ADOLESCENT = "adolescent"
    YOUNG_ADULT = "young_adult"
    MIDDLE_AGED = "middle_aged"
    AGED_80_AND_OVER = "aged_80_and_over"

class OtherFilter(str, Enum):
    EXCLUDE_PREPRINTS = "exclude_preprints"
    MEDLINE = "medline"
    PUBMED_NOT_MEDLINE = "pubmed_not_medline"
    IN_PROCESS = "in_process"
    PUBLISHER = "publisher"
    PMC = "pmc"
    NIHMS = "nihms"

class SortBy(str, Enum):
    RELEVANCE = "relevance"
    PUBLICATION_DATE = "publication_date"
    FIRST_AUTHOR = "first_author"
    LAST_AUTHOR = "last_author"
    JOURNAL = "journal"
    TITLE = "title"

class SearchField(str, Enum):
    TITLE_ABSTRACT = "title/abstract"
    TITLE = "title"
    ABSTRACT = "abstract"
    AUTHOR = "author"
    ALL_FIELDS = "all_fields"