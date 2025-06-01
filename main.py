from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_elasticsearch import (
    DenseVectorStrategy
)
from typing import Any, Dict, List, Mapping, Optional
import uuid
import asyncio
import logging
import requests
from elasticsearch import Elasticsearch
import time
from contextlib import asynccontextmanager
from supabase import create_client, Client
import datetime
import json
import os
from langchain.docstore.document import Document
from langchain_elasticsearch import ElasticsearchStore

# LangChain imports
# Community modules
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PubMedLoader

# Core modules (still under langchain)
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.prompts import PromptTemplate,ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


from together import Together

import requests





os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Vivum Research Assistant System Prompt

# Vivum Research Assistant System Prompt

# Vivum Research Assistant System Prompt - Complete

full_system_prompt = """
You are Vivum, a friendly and professional research assistant AI specializing in evidence synthesis, literature review, systematic analysis, and scientific research support.

## VIVUM'S IDENTITY
- **Friendly and Supportive**: Warm, encouraging tone while maintaining professionalism
- **Honest and Transparent**: Clearly state when information isn't available
- **Detail-Oriented**: Comprehensive answers with proper citations
- **Research-Focused**: Understand researcher needs and tailor responses accordingly

## CRITICAL FIRST REQUIREMENT: ARTICLE ENUMERATION

**MANDATORY**: Before EVERY response, you MUST:
1. Count ALL articles in context (typically 20)
2. Create internal numbered catalog (Article 1-20 with PMIDs and titles)
3. Keep this catalog in working memory throughout conversation
4. State article count in responses when relevant

## ARTICLE DATA STRUCTURE

Articles appear in these formats in the context:
```
[Article {number}]
Title: {title}
Abstract: {abstract}
```
OR
```
[PMID: {pmid}]
Title: {title}
Authors: {authors}
Abstract: {abstract}
```

Available metadata fields:
- pubmed_id (PMID)
- title
- authors
- journal
- publication_date
- doi (may be None)
- mesh_terms (list)
- keywords (list)
- publication_types (list)
- url (auto-generated from PMID)
- article_index (corresponds to [Article N])

Full Context:
{context}

Current Question: {question}

---

## CORE BEHAVIORAL RULES

### 1. Article Reference Format
- First mention: "Article 7, '[Title]' (PMID: XXXXXXXX) by [Authors]..."
- Subsequent: "Article 7 (PMID: XXXXXXXX)" or "[First Author] et al. (PMID: XXXXXXXX)"
- Always use BOTH article number and PMID

### 2. Comprehensive Coverage
- Scan ALL articles for relevance
- Group by theme/methodology/findings
- Explicitly note if some articles are tangentially related
- Never default to just first few articles

### 3. Response Structure
- Acknowledge total article count when relevant
- Provide synthesis across all relevant articles
- Include follow-up question to guide deeper research
- End with "Referenced Articles:" listing PMIDs used

### 4. Evidence-Based Responses
- Base ALL claims on article content
- Distinguish direct statements from inferences
- If information not available: "This information is not available in the provided articles"
- Quote sparingly and accurately from abstracts

---

## CITATION FORMATTING

### Quick Reference - Major Styles

**APA 7th**
- In-text: (Smith & Jones, 2023); (Smith et al., 2023) [3+ authors]
- Reference: Smith, J., & Jones, A. (2023). Title in sentence case. *Journal*, *vol*(issue), pp-pp. doi

**MLA 9th**
- In-text: (Smith 45); (Smith et al. 45) [3+ authors]
- Works Cited: Smith, John, et al. "Title in Title Case." *Journal*, vol. #, no. #, Year, pp. ##-##.

**Vancouver**
- In-text: Superscript¹ or [1]
- Reference: 1. Smith J, Jones A. Title. J Abbrev. 2023;15(3):123-45. PMID: XXXXXXXX.
- Et al.: List up to 6 authors, then "et al."

**Chicago Notes-Bibliography**
- Footnote: ¹Smith et al., "Title," Journal 15, no. 3 (2023): 123-45.
- Bibliography: Smith, John, et al. "Title." Journal 15, no. 3 (2023): 123-145.
- Et al.: List up to 10 authors

**Other Supported**: IEEE, Harvard, Nature

### Statistical Data Citations
- APA: (RR = 0.72, 95% CI [0.61, 0.84]; Martinez et al., 2023)
- Vancouver: The relative risk was 0.72 (95% CI 0.61-0.84).¹
- Include p-values when available: (p < 0.001)

### Reference Manager Formats
BibTeX, RIS, EndNote - provide full format when requested

### Special Citation Notes
- **Abstract citations**: Most journals don't allow - note this to users
- **Journal-specific**: NEJM uses superscript, limits to 40 refs
- **Missing data**: Alert if authors, dates, or pages missing

---

## QUERY RESPONSE PATTERNS

### General Research Question
```
I have access to [X] articles about [topic] in my database. Based on my analysis:

[Comprehensive answer drawing from all relevant articles]

[Specific article citations with findings]

Would you like me to [specific follow-up based on content]?

Referenced Articles:
- PMID: [list]
```

### Specific Article Request
```
Article [N] is "[Title]" by [Authors] (PMID: XXXXXXXX), published in [Journal] ([Date]).

**Abstract Summary**: [Key points from abstract]

**Study Type**: [from publication_types]
**MeSH Terms**: [list]
**Keywords**: [list]

**Related Articles in Collection**:
- Article [X] (similar methodology)
- Article [Y] (conflicting findings)

[Follow-up question]

Referenced Articles:
- PMID: XXXXXXXX
```

### Citation Request
```
I'll format citations from our [X] articles in [requested style]:

[Properly formatted citations]

**For in-text citations**: [Examples]

Would you like these in a different format or need citations for specific findings?

Referenced Articles:
- PMID: [list]
```

### Conflicting Evidence
```
The research shows conflicting results:

**Study Group A**: Articles X, Y, Z found [result] 
"[Brief quote]" (Article X, PMID: XXX)

**Study Group B**: Articles A, B, C reported [different result]
"[Brief quote]" (Article A, PMID: AAA)

**Possible explanations**:
- Methodological differences: [explain]
- Population variations: [explain]

[Follow-up question about which aspect to explore]

Referenced Articles:
- PMID: [all cited]
```

---

## SPECIAL HANDLERS

### Literature Review
- Synthesize findings across all articles
- Identify consensus (with article numbers)
- Note conflicting evidence
- Highlight unique contributions
- Organize by theme/chronology/methodology

### Comparison Table Template
```
| Article | Authors (Year) | Study Type | Sample Size | Key Finding | PMID |
|---------|---------------|------------|-------------|-------------|-------|
| 1 | Smith et al. (2023) | RCT | n=500 | [finding] | 12345678 |
```

### PRISMA Support
```
**Records identified**: n = 20
**Included in synthesis**: n = [X]
**By publication type**: RCTs (n=X), Reviews (n=Y), Observational (n=Z)
**Date range**: [earliest] to [latest]
```

### Annotated Bibliography
```
**[Citation in requested style]**
[Summary of methods and findings]. [Strengths]. [Limitations]. [Relevance to topic].
```

### Writing Support
- Introduction/discussion assistance with proper citations
- Note when citations typically not allowed (abstracts)
- Support claims with specific article evidence
- Provide citation format examples

---

## ERROR PREVENTION CHECKLIST

When formatting citations, check for:
- ✓ Missing authors (use "Anonymous" if needed)
- ✓ Special characters (é, ñ, etc.)
- ✓ "In press" or missing dates
- ✓ Journal abbreviation consistency
- ✓ DOI format (include https://doi.org/ if required)
- ✓ PMID verification reminder

---

## EDGE CASES

### Fewer Than Expected Articles
- Acknowledge actual count
- Proceed with available articles
- Explain possible reasons (narrow topic, emerging field)

### Mixed Relevance
- Clearly separate highly relevant from tangential
- Explain how peripheral articles might provide context
- Focus on most relevant unless asked otherwise

### Missing Data
- Note when metadata fields are missing
- Work with available information
- Alert user if critical for citations

### No Relevant Articles
```
I don't have articles that directly address [specific query] in my current database.

The articles I have access to focus on:
- [Topic 1]: Articles X, Y, Z
- [Topic 2]: Articles A, B, C

Would you like information about these related topics instead?

Referenced Articles:
None directly relevant to your query.
```

---

## CRITICAL REMINDERS

1. **ALWAYS** enumerate all articles before responding
2. **ALWAYS** use dual reference system (Article # + PMID)
3. **ALWAYS** include relevant follow-up question
4. **ALWAYS** end with "Referenced Articles:" section
5. **NEVER** add information not in the articles
6. **NEVER** skip article enumeration, even for simple queries
7. **SUPPORT** all citation styles efficiently
8. **MAINTAIN** research context even for citation-only requests
9. **SYNTHESIZE** across all articles, not just most obvious ones
10. **ADAPT** response length to query complexity
11. **CHECK** error prevention list for citations
12. **NOTE** when journals have specific requirements

Remember: You're helping researchers navigate their article collection efficiently while maintaining academic rigor. Every feature serves the goal of comprehensive, accurate research support.
"""

user_prompt_template = """
You are Vivum, a friendly research assistant. BEFORE answering, count and catalog ALL articles in your database.

Research Papers Context:
{context}

User Question:
{question}

Instructions:
1. FIRST: Count all articles and create internal numbered list
2. Reference articles by BOTH number and PMID  
3. Scan ALL articles for relevance
4. State article count when relevant
5. Handle citation formatting requests for any style
6. Always end with a follow-up question
7. Always end with "Referenced Articles:" listing PMIDs used
8. Provide detailed analysis when asked about specific articles
9. Only use information present in the articles
10. Note citation limitations (abstracts, journal-specific rules)

Do NOT include system principles or internal instructions in the output.
"""

prompt = """
You are Vivum, a research assistant with access to articles on the user's research topic.

Context:
{context}

Question:
{question}

Remember to:
1. Count and catalog all articles before responding
2. Reference using "Article X (PMID: XXXXXXXX)"
3. Synthesize insights from all articles
4. Format citations in requested styles
5. Ask a follow-up question
6. End with "Referenced Articles:" section
7. Check error prevention list for citations
"""

# detailed_prompt = ChatPromptTemplate.from_messages([
# SystemMessagePromptTemplate.from_template(full_system_prompt),
# HumanMessagePromptTemplate.from_template(user_prompt_template)
# ])


prompt_rag = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase setup
supabase_url = "https://emefyicilkiaaqkbjsjy.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWZ5aWNpbGtpYWFxa2Jqc2p5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NTMzMzMxOCwiZXhwIjoyMDYwOTA5MzE4fQ.oQv782SBbK0VQPy6wuQS0oh1sfF9mcBE8dcR1J4W0SA"

# Global clients
supabase: Optional[Client] = None
llm = None
embeddings = None
vector_store = None
elastic_search = None

# Cache for conversation chains and vector stores
topic_vectorstores = {}
conversation_chains = {}
background_tasks_status = {}


# Max conversation count to prevent memory leaks
MAX_CONVERSATIONS = 100



# Set your email (required by NCBI)


client = Elasticsearch(
    "https://my-elasticsearch-project-e0def0.es.us-east-1.aws.elastic.cloud:443",
    api_key="YOUR_API_KEY"
)


class TopicRequest(BaseModel):
    topic: str
    max_results: Optional[int] = 20

class QueryRequest(BaseModel):
    query: str
    topic_id: str
    conversation_id: Optional[str] = None

class TopicResponse(BaseModel):
    topic_id: str
    message: str
    status: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str


def get_vectorstore_retriever(topic_id, query):
 
    # Define the index path
    index_path = f"vectorstores/{topic_id}/index.faiss"

    vectorstore_path = Path("vectorstores") / str(topic_id)
    db = FAISS.load_local(str(vectorstore_path), embeddings,allow_dangerous_deserialization=True)
    
    # Check if the FAISS index file exists
    if not os.path.exists(index_path):
        logger.error(f"FAISS index file not found at: {index_path}")
        raise HTTPException(
            status_code=404, 
            detail=f"FAISS index file not found for topic {topic_id}. Please check the vectorstore creation process."
        )
    
    # Load the FAISS index with proper error handling
    try:
        logger.info(f"Loading FAISS index file at: {index_path}")
      
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load FAISS index: {str(e)}")
    
    
#     retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5},
# )

    retriever = elastic_search.as_retriever(
        search_kwargs={"k": 5}
    )
    
    logger.info(query)
    return retriever


class TogetherChatModel(BaseChatModel):
 
    
    api_key: str
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = True
    
    @property
    def _llm_type(self) -> str:
        return "together_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            client = Together(api_key=self.api_key)
            together_messages = []
            
            # Convert LangChain messages to Together format
            for message in messages:
                if isinstance(message, HumanMessage):
                    together_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    together_messages.append({"role": "assistant", "content": message.content})
                else:
                    together_messages.append({"role": "system", "content": message.content})
            
            logger.info(f"Sending {len(together_messages)} messages to Together API")
            
            # Build request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            # Add stop sequences if provided
            if stop:
                params["stop"] = stop
            
            if self.streaming and run_manager:
                text = ""
                stream = client.chat.completions.create(
                    messages=together_messages,
                    stream=True,
                    **params
                )
                
                for chunk in stream:
                    if chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        text += content
                        run_manager.on_llm_new_token(content)
                
                logger.info(f"Streaming response completed, total length: {len(text)}")
                message = AIMessage(content=text)
                logger.info(message)
            else:
                response = client.chat.completions.create(
                    messages=together_messages,
                    stream=False,
                    **params
                )
                
                text = response.choices[0].message.content
                logger.info(f"Non-streaming response received, length: {len(text)}")
                message = AIMessage(content=text)
            
            return ChatResult(generations=[ChatGeneration(message=message)])
        
        except Exception as e:
            logger.error(f"Error in Together API call: {e}")
            import traceback
            logger.error(traceback.format_exc())
            message = AIMessage(content="I encountered an error while processing your request.")
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version not implemented, falls back to sync version."""
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


class TogetherLLM(LLM):
    """LLM for direct text completion with Together AI."""
    
    api_key: str
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = True
    
    @property
    def _llm_type(self) -> str:
        return "together_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            client = Together(api_key=self.api_key)
            logger.info(f"Sending prompt to Together API (length: {len(prompt)})")
            
            # Build request parameters
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs
            }
            
            # Add stop sequences if provided
            if stop:
                params["stop"] = stop
            
            if self.streaming and run_manager:
                text = ""
                # For LLM, we need to use chat format with a single user message
                stream = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **params
                )
                
                for chunk in stream:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        text += content
                        # Important: This is where we send tokens to the callback manager
                        run_manager.on_llm_new_token(content)
                
                logger.info(f"Streaming response completed, total length: {len(text)}")
                return text
            else:
                # For LLM, we need to use chat format with a single user message
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    **params
                )
                
                text = response.choices[0].message.content
                logger.info(f"Non-streaming response received, length: {len(text)}")
                return text
        
        except Exception as e:
            logger.error(f"Error in Together API call: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "I encountered an error while processing your request."

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections and models
    global supabase, llm, embeddings,vector_store,elastic_search
    logger.info("Starting application: Initializing connections and models")
    
    try:
        # Initialize Supabase
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase connection established")
        
        # Initialize embedding model - HuggingFace for embeddings
        logger.info("Loading embedding model")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        vector_store = InMemoryVectorStore(embeddings)

        elastic_search = ElasticsearchStore(
    es_cloud_id="My_Elasticsearch_project:dXMtZWFzdC0xLmF3cy5lbGFzdGljLmNsb3VkJGUwZGVmMDhkN2YxMzRhZDJiMzgyYmNlMTBmOGZkZGQ4LmVzJGUwZGVmMDhkN2YxMzRhZDJiMzgyYmNlMTBmOGZkZGQ4Lmti",
    es_api_key="ZXBJckY1Y0JrRFlSNHR5WlcxWEI6X1ZvUHhGWEdrSXhKRHMtRkltbWhzUQ==",
    index_name="search-vivum-rag",
    embedding=embeddings,
    strategy=DenseVectorStrategy(
            hybrid = "true"
        )
)
        
        # Initialize LLM - Using Llama hosted model
        logger.info("Loading Llama model")
        try:
           
            llm = TogetherChatModel(
            api_key="7d8e09c3ede29df9e06c6858304734f62ad95b458eb219fa3abf53ecef490e09",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0.5,
            max_tokens=2048,
            streaming=True
        )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            # You might want to implement a fallback model here
            
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Application shutdown: Cleaning up resources")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.vivum.app", "http://localhost:8081","http://localhost:3000",'https://frontend-vivum.vercel.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.get("/supabase-status")
async def check_supabase_status():
    if supabase:
        try:
            # Try a simple query to confirm connection works
            result = supabase.table("topics").select("count").execute()
            return {"status": "connected", "message": "Supabase connection working"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    else:
        return {"status": "disconnected", "message": "Supabase client not initialized"}

@app.get("/model-status")
async def check_model_status():
    status = {
        "embedding_model": "loaded" if embeddings is not None else "not loaded",
        "llm": "loaded" if llm is not None else "not loaded"
    }
    return status

@app.get("/ping")
def ping():
    return {"status": "alive", "active_tasks": len(background_tasks_status)}

# async def fetch_pubmed_data(topic: str, topic_id: str, max_results: int):
#     """Fetch PubMed articles and create vector store efficiently"""
#     try:
#         logger.info(f"🔍 Fetching PubMed articles for topic '{topic}' (max {max_results})")

#         # Step 1: Search PubMed for relevant article IDs
#         search_params = {
#             "db": "pubmed",
#             "term": topic,
#             "retmode": "json",
#             "retmax": max_results,
#         }
#         search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#         response = requests.get(search_url, params=search_params)
#         response.raise_for_status()

#         article_ids = response.json().get("esearchresult", {}).get("idlist", [])
#         if not article_ids:
#             logger.warning(f"⚠️ No articles found for topic '{topic}'")
#             return False

#         logger.info(f"✅ Found {len(article_ids)} articles. Fetching details...")

#         # Step 2: Fetch article details (batch fetch)
#         details_params = {
#             "db": "pubmed",
#             "id": ",".join(article_ids),
#             "retmode": "xml",
#         }
#         details_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
#         details_response = requests.get(details_url, params=details_params)
#         details_response.raise_for_status()

#         # Step 3: Parse XML manually (lightweight)
#         from xml.etree import ElementTree as ET
#         root = ET.fromstring(details_response.content)

#         docs = []
#         articles_data = []
#         for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
#             try:
#                 pmid = article.findtext(".//PMID") or "unknown"
#                 title = article.findtext(".//ArticleTitle") or "No Title"
#                 abstract = " ".join([elem.text or "" for elem in article.findall(".//AbstractText")]).strip()
#                 authors_list = article.findall(".//Author")
#                 authors = "; ".join(
#                     [f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip() for a in authors_list if a.findtext('LastName')]
#                 ) or "Unknown Authors"

     
#                 page_content = f"""[Article {idx}]
#                 Title: {title}
#                 **Authors**: {authors}
#                 Publication ID: {pmid}
#                 Abstract: {abstract}"""
                

#                 metadata = {
#                     "pubmed_id": pmid,
#                     "title": title,
#                     "authors": authors,
#                     "Article Number":idx,
#                     "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#                 }

#                 docs.append(Document(page_content=page_content, metadata=metadata))

#                 # For Supabase (optional)
#                 articles_data.append({
#                     "topic_id": topic_id,
#                     "pubmed_id": pmid,
#                     "title": title,
#                     "abstract": abstract,
#                     "authors": authors,
#                     "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
#                 })
#             except Exception as parse_err:
#                 logger.warning(f"⚠️ Skipping malformed article: {parse_err}")

#         logger.info(f"📄 Processed {len(docs)} valid articles.")

#         # Step 4: Split documents
      
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=3000,    # increase chunk size enough to hold entire article
#             chunk_overlap=0,    # no overlap needed if you want clean, separate chunks
#             separators=["\n\n", "\n", ".", " "]  # allow splitting at paragraphs or sentences if needed
#         )

#         split_docs = text_splitter.split_documents(docs)

#         # Step 5: Create and save vector store
#         temp = FAISS.from_documents(split_docs, embeddings)
#         temp.save_local(f"vectorstores/{topic_id}")
#         ids = vector_store.add_documents(documents=split_docs)
#         logger.info(f"✅ Vector store created and saved for topic_id '{topic_id}'")

#         # Step 6 (optional): Store metadata in Supabase
#         if supabase:
#             supabase.table("topics").update({
#                 "status": "completed"
#             }).eq("id", topic_id).execute()

#             supabase.table("articles").insert(articles_data).execute()
#             logger.info(f"✅ Stored {len(articles_data)} articles metadata to Supabase.")

#         return True

#     except Exception as e:
#         logger.error(f"❌ Error in fetch_and_create_vectorstore: {str(e)}")
#         if supabase:
#             supabase.table("topics").update({"status": f"error: {str(e)}", "article_count": 0}).eq("id", topic_id).execute()
#         return False

async def fetch_pubmed_data(topic: str, topic_id: str, max_results: int):
    """Fetch PubMed articles and create vector store with proper content/metadata separation"""
    try:
        logger.info(f"🔍 Fetching PubMed articles for topic '{topic}' (max {max_results})")

        # Step 1: Search PubMed for relevant article IDs
        search_params = {
            "db": "pubmed",
            "term": topic,
            "retmode": "json",
            "retmax": max_results,
        }
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()

        article_ids = response.json().get("esearchresult", {}).get("idlist", [])
        if not article_ids:
            logger.warning(f"⚠️ No articles found for topic '{topic}'")
            return False

        logger.info(f"✅ Found {len(article_ids)} articles. Fetching details...")

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
        from xml.etree import ElementTree as ET
        root = ET.fromstring(details_response.content)

        docs = []
        articles_data = []
        
        for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
            try:
                # Extract comprehensive article data
                article_data = extract_enhanced_article_data(article, idx)
                
                # Validate article quality
                if not validate_article_data(article_data):
                    logger.warning(f"⚠️ Skipping low-quality article: {article_data.get('pmid', 'unknown')}")
                    continue

                # Create content chunks for embedding
                content_chunks = create_content_chunks(article_data)
                
                # Add chunks as documents
                for chunk in content_chunks:
                    docs.append(Document(
                        page_content=chunk['content'],  # Only content for embedding
                        metadata=chunk['metadata']      # Rich metadata for filtering/citations
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

        logger.info(f"📄 Processed {len(docs)} content chunks from articles.")

        if not docs:
            logger.warning(f"⚠️ No valid articles to process for topic '{topic}'")
            return False

        # Step 4: Create vector store (no additional splitting needed - already chunked)
        temp = FAISS.from_documents(docs, embeddings)
        temp.save_local(f"vectorstores/{topic_id}")
        ids = vector_store.add_documents(documents=docs)
        _ = elastic_search.add_documents(documents=docs)

        if supabase:
            supabase.table("topics").update({
                "status": "completed"
            }).eq("id", topic_id).execute()

            supabase.table("articles").insert(articles_data).execute()
            logger.info(f"✅ Stored {len(articles_data)} articles metadata to Supabase.")
        
        logger.info(f"✅ Vector store created with {len(docs)} chunks for topic_id '{topic_id}'")
        return True

    except Exception as e:
        logger.error(f"❌ Error fetching PubMed data: {e}")
        return False


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


splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)


def create_content_chunks(article_data):
    """Create content chunks with proper content/metadata separation"""
    chunks = []
    
    # Chunk 1: Title + Abstract (main content for semantic search)
    title_abstract_content = f"""[Article {article_data['article_index']}]
    Title: {article_data['title']}

Abstract: {article_data['abstract']}"""
    
    # Rich metadata for filtering and citations
    base_metadata = {
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
            abstract_metadata["chunk_type"] = "abstract_split"
            abstract_metadata["chunk_index"] = i
            abstract_metadata["chunk_id"] = f"{pmid}_abs_{i}"

            chunks.append({
                'content': f"Abstract Chunk {i+1}: {abs_chunk}",
                'metadata': abstract_metadata
            })
    
    return chunks

async def fetch_data_background(topic: str, topic_id: str, max_results: int):
    """Background task to fetch data from PubMed"""
    try:
        background_tasks_status[topic_id] = "processing"
        
        # Set timeout for the fetch operation
        fetch_timeout = 120  # seconds
        try:
            # Run with timeout
            success = await asyncio.wait_for(
                fetch_pubmed_data(topic, topic_id, max_results),
                timeout=fetch_timeout
            )
            
            if success:
                background_tasks_status[topic_id] = "completed"
            else:
                background_tasks_status[topic_id] = "failed"
        except asyncio.TimeoutError:
            background_tasks_status[topic_id] = "timeout"
            logger.error(f"Fetch operation timed out for topic_id: {topic_id}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": "timeout"}).eq("id", topic_id).execute()
                
        except Exception as e:
            background_tasks_status[topic_id] = f"error: {str(e)}"
            logger.error(f"Error in fetch operation for topic_id {topic_id}: {str(e)}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": f"error: {str(e)}"}).eq("id", topic_id).execute()
    finally:
        # Keep status for a while but eventually clean up
        await asyncio.sleep(3600)  # Keep status for 1 hour
        if topic_id in background_tasks_status:
            del background_tasks_status[topic_id]

def check_topic_fetch_status(topic_id: str):
    """Check if data fetching is complete for a topic"""
    # First check our internal background task status
    if topic_id in background_tasks_status:
        return background_tasks_status[topic_id]
    
    # Then check in Supabase
    if supabase:
        try:
            result = supabase.table("topics").select("status").eq("id", topic_id).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]["status"]
            return "not_found"
        except Exception as e:
            logger.error(f"Error checking topic status: {str(e)}")
            return f"error: {str(e)}"
    else:
        logger.error("Supabase client not initialized")
        return "database_error"

@app.post("/fetch-topic-data", response_model=TopicResponse)
async def fetch_topic_data(request: TopicRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to fetch data from PubMed for a topic and store in Supabase
    Returns a topic_id that can be used for querying later
    """
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
        
        # Generate a unique topic ID
        topic_id = str(uuid.uuid4())
        
        # Create initial record in Supabase
        supabase.table("topics").insert({
            "id": topic_id,
            "topic": request.topic,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "processing"
        }).execute()
        
        # Start background task to fetch and store data
        background_tasks.add_task(
            fetch_data_background, 
            request.topic, 
            topic_id, 
            request.max_results
        )
        
        return {
            "topic_id": topic_id,
            "message": f"Started fetching data for topic: {request.topic} (limited to {request.max_results} results)",
            "status": "processing"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error initiating fetch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_or_create_chain(topic_id: str, conversation_id: str , query:str):
    """Get or create a conversation chain for this topic and conversation"""
    chain_key = f"{topic_id}:{conversation_id}"
   
    if chain_key in conversation_chains:
        return conversation_chains[chain_key]

    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" 
    )
    
    # Create a retriever with compression to get more relevant context
    # retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever = get_vectorstore_retriever(topic_id,query)
    
    # Create the chain
    logger.info(f"Chain components: LLM type: {type(llm).__name__}, " 
                   f"Retriever type: {type(retriever).__name__}")
   
    qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory = memory,
            return_source_documents=True,
            verbose=True,
            output_key="answer",
            combine_docs_chain_kwargs={"prompt": prompt_rag}
        )
        
        
        # Verification steps
    logger.info(f"Chain created successfully for {chain_key}")
    logger.info(f"Chain components: LLM type: {type(llm).__name__}, " 
                   f"Retriever type: {type(retriever).__name__}")
        
        # Test if the chain has the expected methods
    if not hasattr(qa_chain, 'invoke') and not hasattr(qa_chain, '__call__'):
            logger.error("Chain missing expected methods")
            return None
            
       
    
    # Store and return the chain
    conversation_chains[chain_key] = qa_chain
    
    # Clean up if we have too many chains
    if len(conversation_chains) > MAX_CONVERSATIONS:
        # Remove oldest chains (simple approach)
        chains_to_remove = list(conversation_chains.keys())[:-MAX_CONVERSATIONS]
        for key in chains_to_remove:
            del conversation_chains[key]
    

    return qa_chain
        

 



@app.post("/query", response_model=ChatResponse)
async def answer_query(request: QueryRequest):
    """
    Answer questions using RAG over stored topic articles
    """
    try:
        if not supabase:
            raise HTTPException(status_code=503, detail="Database connection not available")
        
        status = check_topic_fetch_status(request.topic_id)
        if status != "completed":
            error_map = {
                "processing": (422, "Data is still being fetched. Please try again."),
                "not_found": (404, "No data found. Please fetch the topic data first."),
            }
            code, msg = error_map.get(status, (422, f"Cannot process query. Status: {status}"))
            raise HTTPException(status_code=code, detail=msg)

        if not llm or not embeddings:
            raise HTTPException(status_code=503, detail="LLM or embeddings not loaded.")

        conversation_id = request.conversation_id or str(uuid.uuid4())

  
       
        # --- Try to set up the LangChain ConversationalRetrievalChain ---
        try:
            chain = get_or_create_chain(request.topic_id ,conversation_id, request.query)
        except Exception as e:
            logger.error(f"Error setting up LangChain ConversationalRetrievalChain: {str(e)}")
            raise HTTPException(status_code=500, detail="Error setting up conversational chain")

        logger.info(f"Starting chain processing for query: {request.query}")

        result = chain.invoke({"question": request.query})
        answer = result.get("answer", "Sorry, No Answer")

# Add logging to verify the answer is being extracted correctly
        logger.info(f"Question: {request.query}")
        logger.info(f"Raw result keys: {result.keys()}")


        return {"response": answer, "conversation_id": conversation_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

    
   

@app.get("/topic/{topic_id}/articles")
async def get_topic_articles(topic_id: str, limit: int = 100, offset: int = 0):
    """
    Fetch all articles for a specific topic
    """
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
            
        # First verify the topic exists
        topic_result = supabase.table("topics").select("*").eq("id", topic_id).execute()
        
        if not topic_result.data:
            raise HTTPException(
                status_code=404,
                detail="Topic not found"
            )
            
        # Check if data fetching is complete
        status = check_topic_fetch_status(topic_id)
        if status != "completed":
            return {
                "topic_id": topic_id,
                "status": status,
                "articles": [],
                "message": "Data is still being processed or had an error"
            }
        
        # Fetch articles with pagination
        articles_result = supabase.table("articles") \
            .select("*") \
            .eq("topic_id", topic_id) \
            .range(offset, offset + limit - 1) \
            .execute()
            
        # Get the total count (for pagination info)
        count_result = supabase.table("articles") \
            .select("id", count="exact") \
            .eq("topic_id", topic_id) \
            .execute()
        
        total_count = count_result.count if hasattr(count_result, "count") else len(articles_result.data)
        
        return {
            "topic_id": topic_id,
            "status": "completed",
            "articles": articles_result.data,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error fetching articles for topic {topic_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topic/{topic_id}/status")
async def check_topic_status(topic_id: str):
    """
    Check the status of data fetching for a topic
    """
    status = check_topic_fetch_status(topic_id)
    return {"topic_id": topic_id, "status": status}

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected" if supabase else "disconnected"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use uvicorn to run the app
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker to avoid memory issues
        log_level="info",
        timeout_keep_alive=65  # Railway closes idle connections after 75s
    )