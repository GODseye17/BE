"""
PubMed Data Fetcher
"""
import asyncio
import logging
import requests
from typing import List, Optional, Dict, Any
from xml.etree import ElementTree as ET
from langchain.docstore.document import Document

from .filters import PubMedFilters
from .article_processor import (
    extract_enhanced_article_data, validate_article_data,
    create_content_chunks
)
from vectorstore.manager import create_faiss_store_in_batches
from core.globals import get_globals

logger = logging.getLogger(__name__)

async def fetch_pubmed_data(topics: Optional[List[str]] = None, operator: str = 'AND',
                           topic: Optional[str] = None, topic_id: str = "", max_results: int = 100,
                           filters: Optional[Dict[str, Any]] = None, 
                           advanced_query: Optional[str] = None):
    """Enhanced PubMed data fetcher with multi-topic boolean search support"""
    try:
        # Get global instances
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        # Initialize filter builder
        filter_builder = PubMedFilters()
        
        # Build complete search query with multi-topic support
        search_query = filter_builder.build_complete_query(
            topics=topics,
            operator=operator,
            base_query=topic,  # For backward compatibility
            filters=filters,
            advanced_query=advanced_query
        )
        
        # Log search details
        if topics:
            logger.info(f"🔍 Multi-topic search for: {topics}")
            logger.info(f"🔗 Boolean operator: {operator}")
        elif topic:
            logger.info(f"🔍 Single topic search for: '{topic}'")
        elif advanced_query:
            logger.info(f"🔍 Advanced query search")
        
        logger.info(f"🔍 Final search query: {search_query}")
        logger.info(f"📊 Max results: {max_results}")

        # Step 1: Search PubMed for relevant article IDs
        search_params = {
            "db": "pubmed",
            "term": search_query,
            "retmode": "json",
            "retmax": max_results,
            "sort": filters.get('sort_by', 'relevance') if filters else 'relevance',
            "field": filters.get('search_field', 'title/abstract') if filters else 'title/abstract'
        }
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()

        search_result = response.json().get("esearchresult", {})
        article_ids = search_result.get("idlist", [])
        total_count = search_result.get("count", "0")
        
        if not article_ids:
            search_description = f"topics {topics} with operator '{operator}'" if topics else f"topic '{topic}'"
            logger.warning(f"⚠️ No articles found for {search_description} with applied filters")
            logger.info(f"📊 Total available articles: {total_count}")
            return False

        logger.info(f"✅ Found {len(article_ids)} articles (total available: {total_count}). Fetching details...")

        if len(article_ids) > 20:
            logger.info(f"⏳ Fetching {len(article_ids)} articles - this may take 30-60 seconds...")

        # Step 2: Fetch article details (batch fetch)
        details_params = {
            "db": "pubmed",
            "id": ",".join(article_ids),
            "retmode": "xml",
        }
        details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        details_response = requests.get(details_url, params=details_params)
        details_response.raise_for_status()

        # Step 3: Parse XML with enhanced extraction
        root = ET.fromstring(details_response.content)

        docs = []
        articles_data = []
        
        for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
            try:
                # Add progress logging every 10 articles
                if idx % 10 == 0:
                    logger.info(f"📄 Processing article {idx}/{len(root.findall('.//PubmedArticle'))}...")
                
                # Extract comprehensive article data
                article_data = extract_enhanced_article_data(article, idx)
                
                # Validate article quality
                if not validate_article_data(article_data):
                    logger.warning(f"⚠️ Skipping low-quality article: {article_data.get('pmid', 'unknown')}")
                    continue

                # Create content chunks for embedding
                content_chunks = create_content_chunks(article_data, topic_id)
                
                # Add chunks as documents
                for chunk in content_chunks:
                    docs.append(Document(
                        page_content=chunk['content'],
                        metadata=chunk['metadata']
                    ))

                articles_data.append({
                    "topic_id": topic_id,
                    "pubmed_id": article_data['pmid'],
                    "title": article_data['title'],
                    "abstract": article_data['abstract'],
                    "authors": article_data['authors'],
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_data['pmid']}/"
                })

            except Exception as parse_err:
                logger.warning(f"⚠️ Skipping malformed article: {parse_err}")

        logger.info(f"📄 Processed {len(docs)} content chunks from {len(articles_data)} articles.")

        if not docs:
            search_description = f"topics {topics}" if topics else f"topic '{topic}'"
            logger.warning(f"⚠️ No valid articles to process for {search_description}")
            return False

        # Step 4: Create vector store with batching for better performance
        try:
            # Use batch processing for faster embedding creation
            db = create_faiss_store_in_batches(docs, topic_id, batch_size=10)
            logger.info(f"✅ Created topic-specific FAISS store for topic {topic_id} with {len(docs)} chunks")
        except Exception as e:
            logger.error(f"❌ Vector store error: {e}")
            raise  # Re-raise to mark the fetch as failed

        # Step 5: Store metadata to Supabase
        try:
            if supabase:
                supabase.table("topics").update({
                    "status": "completed",
                    "total_articles_found": total_count,
                    "article_count": len(articles_data),
                    "search_topics": topics,
                    "boolean_operator": operator,
                    "filters": filters
                }).eq("id", topic_id).execute()

                supabase.table("articles").insert(articles_data).execute()
                logger.info(f"✅ Stored {len(articles_data)} articles metadata to Supabase.")
        except Exception as e:
            logger.warning(f"⚠️ Supabase error: {e}")
        
        logger.info(f"✅ Processing completed with {len(docs)} chunks for topic_id '{topic_id}'")
        return True

    except Exception as e:
        logger.error(f"❌ Error fetching PubMed data: {e}")
        try:
            if supabase:
                supabase.table("topics").update({
                    "status": f"error: {str(e)}", 
                    "article_count": 0
                }).eq("id", topic_id).execute()
        except:
            pass
        return False