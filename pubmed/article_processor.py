"""
Article Processing Functions for PubMed Data
"""
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Initialize text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

def extract_enhanced_article_data(article, idx):
    """Extract comprehensive article data with proper error handling"""
    # Basic identifiers
    pmid = article.findtext(".//PMID") or "unknown"
    
    # Title
    title = article.findtext(".//ArticleTitle") or "No Title"
    
    # Enhanced abstract extraction (handles structured abstracts)
    abstract = extract_structured_abstract(article)
    
    # Enhanced author extraction
    authors = extract_authors_with_affiliations(article)
    
    # Journal information
    journal = (article.findtext(".//Journal/Title") or 
              article.findtext(".//MedlineTA") or 
              "Unknown Journal")
    
    # Publication date
    publication_date = extract_publication_date(article)
    
    # DOI extraction
    doi = extract_doi(article)
    
    # MeSH terms and keywords
    mesh_terms = extract_mesh_terms(article)
    keywords = extract_keywords(article)
    
    # Publication types
    pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
    
    return {
        'article_index': idx,
        'pmid': pmid,
        'title': title,
        'abstract': abstract,
        'authors': authors,
        'journal': journal,
        'publication_date': publication_date,
        'doi': doi,
        'mesh_terms': mesh_terms,
        'keywords': keywords,
        'publication_types': pub_types
    }

def extract_structured_abstract(article):
    """Extract abstract handling structured formats with labels"""
    abstract_texts = article.findall(".//AbstractText")
    if not abstract_texts:
        return "No Abstract"
    
    abstract_parts = []
    for elem in abstract_texts:
        label = elem.get('Label', '')
        text = elem.text or ''
        if text.strip():
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
    
    return " ".join(abstract_parts).strip() or "No Abstract"

def extract_authors_with_affiliations(article):
    """Extract authors with proper name formatting"""
    authors = []
    for author in article.findall(".//Author"):
        if author.find("CollectiveName") is not None:
            collective_name = author.findtext("CollectiveName", "").strip()
            if collective_name:
                authors.append(collective_name)
        else:
            last = author.findtext("LastName", "").strip()
            first = author.findtext("ForeName", "").strip()
            initials = author.findtext("Initials", "").strip()
            
            if last:
                if first:
                    full_name = f"{last}, {first}"
                elif initials:
                    full_name = f"{last}, {initials}"
                else:
                    full_name = last
                authors.append(full_name)
    
    return "; ".join(authors) or "Unknown Authors"

def extract_publication_date(article):
    """Extract publication date from various date fields"""
    date_elem = (article.find(".//PubDate") or 
                article.find(".//ArticleDate") or 
                article.find(".//DateCompleted"))
    
    if date_elem is not None:
        year = date_elem.findtext("Year", "")
        month = date_elem.findtext("Month", "")
        day = date_elem.findtext("Day", "")
        
        if year:
            try:
                # Convert month name to number if needed
                if month and not month.isdigit():
                    month_map = {
                        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                    }
                    month = month_map.get(month[:3], month)
                
                if month and day:
                    return f"{year}-{str(month).zfill(2)}-{day.zfill(2)}"
                elif month:
                    return f"{year}-{str(month).zfill(2)}"
                else:
                    return year
            except:
                return year
    
    return "Unknown Date"

def extract_doi(article):
    """Extract DOI from article identifiers"""
    for article_id in article.findall(".//ArticleId"):
        if article_id.get("IdType") == "doi":
            return article_id.text
    return None

def extract_mesh_terms(article):
    """Extract MeSH terms for topic classification"""
    mesh_terms = []
    for mesh in article.findall(".//MeshHeading/DescriptorName"):
        if mesh.text:
            mesh_terms.append(mesh.text)
    return mesh_terms

def extract_keywords(article):
    """Extract author keywords"""
    keywords = []
    for keyword in article.findall(".//Keyword"):
        if keyword.text:
            keywords.append(keyword.text)
    return keywords

def validate_article_data(article_data):
    """Validate article quality before processing"""
    # Must have PMID
    if not article_data.get('pmid') or article_data['pmid'] == 'unknown':
        return False
    
    # Must have meaningful title
    title = article_data.get('title', '')
    if not title or title == 'No Title' or len(title.strip()) < 10:
        return False
    
    # Must have abstract or be a substantial title
    abstract = article_data.get('abstract', '')
    if (not abstract or abstract == 'No Abstract' or len(abstract.strip()) < 50) and len(title) < 50:
        return False
    
    return True

def create_content_chunks(article_data, topic_id):
    """Create content chunks with proper content/metadata separation"""
    chunks = []
    
    # Rich metadata for filtering and citations
    base_metadata = {
        "topic_id": topic_id,  # CRITICAL: Add this line
        "pubmed_id": article_data.get('pmid', 'unknown'),
        "title": article_data.get('title', 'No Title'),
        "authors": article_data.get('authors', 'Unknown Authors'),
        "journal": article_data.get('journal', 'Unknown Journal'),
        "publication_date": article_data.get('publication_date', 'Unknown Date'),
        "doi": article_data.get('doi'),  # Can be None
        "mesh_terms": article_data.get('mesh_terms', []),
        "keywords": article_data.get('keywords', []),
        "publication_types": article_data.get('publication_types', []),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_data.get('pmid', 'unknown')}/",
        "chunk_type": "title_abstract"
    }

    # Safe content creation
    title = base_metadata["title"]
    authors = base_metadata["authors"]
    abstract = article_data.get('abstract', 'No Abstract')
    pmid = base_metadata["pubmed_id"]
    
    title_abstract_content = f"""[PMID: {pmid}]
Title: {title}
Authors: {authors}

Abstract: {abstract}"""
    
    chunks.append({
        'content': title_abstract_content,
        'metadata': base_metadata
    })
    
    # If abstract is very long, create additional chunk for just abstract
    if len(abstract) > 800:
        abstract_chunks = splitter.split_text(abstract)
        for i, abs_chunk in enumerate(abstract_chunks):
            
            abstract_metadata = base_metadata.copy()
            abstract_metadata["topic_id"] = topic_id  # ENSURE topic_id is preserved
            abstract_metadata["chunk_type"] = "abstract_split"
            abstract_metadata["chunk_index"] = i
            abstract_metadata["chunk_id"] = f"{pmid}_abs_{i}"

            chunks.append({
                'content': f"Abstract Chunk {i+1}: {abs_chunk}",
                'metadata': abstract_metadata
            })
    
    return chunks