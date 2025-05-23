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
from typing import Any, Dict, List, Mapping, Optional
import uuid
import asyncio
import logging
import requests
import time
from contextlib import asynccontextmanager
from supabase import create_client, Client
import datetime
import json
import os
from langchain.docstore.document import Document

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

from langchain.prompts import PromptTemplate

from together import Together




os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Create a more constrained prompt that handles missing information better
detailed_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert research assistant helping users find information from scientific papers.

Research Papers (each begins with a label like [Article 1], [Article 2], etc.):
{context}

Chat History and Current Question: {question}

Guidelines:
1. Each paper is labeled with an article number like [Article 1], [Article 2], etc. Always refer to these numbers and their corresponding abstract when discussing specific papers.
2. When answering questions about authors, look carefully through the provided context for author names.
3. When discussing papers, include relevant details like PubMed IDs when available.
4. If asked "who wrote" or "who is the author", search through all the provided papers for author information.
5. If the question mentions "article 2", "article 3", etc., find the paper with that exact article number and use it's abstract to answer questions.
6. Be specific about which paper you're referring to when multiple papers are mentioned.
7. If you cannot find the requested information in the provided context, clearly state this.

Please extract your answer directly from the research papers above. If a specific article number is mentioned (e.g., [Article 3]), find the exact article with that label and extract details like title, authors, abstract, and PubMed ID directly from that section"""
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

# Cache for conversation chains and vector stores
topic_vectorstores = {}
conversation_chains = {}
background_tasks_status = {}


# Max conversation count to prevent memory leaks
MAX_CONVERSATIONS = 100

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
    
    
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
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
    global supabase, llm, embeddings,vector_store
    logger.info("Starting application: Initializing connections and models")
    
    try:
        # Initialize Supabase
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase connection established")
        
        # Initialize embedding model - HuggingFace for embeddings
        logger.info("Loading embedding model")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = InMemoryVectorStore(embeddings)
        
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

async def fetch_pubmed_data(topic: str, topic_id: str, max_results: int):
    """Fetch PubMed articles and create vector store efficiently"""
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

        # Step 3: Parse XML manually (lightweight)
        from xml.etree import ElementTree as ET
        root = ET.fromstring(details_response.content)

        docs = []
        articles_data = []
        for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
            try:
                pmid = article.findtext(".//PMID") or "unknown"
                title = article.findtext(".//ArticleTitle") or "No Title"
                abstract = " ".join([elem.text or "" for elem in article.findall(".//AbstractText")]).strip()
                authors_list = article.findall(".//Author")
                authors = "; ".join(
                    [f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip() for a in authors_list if a.findtext('LastName')]
                ) or "Unknown Authors"

                # page_content = f"Title: {title}\nAbstract: {abstract}"
                # metadata = {
                #     "pubmed_id": pmid,
                #     "title": title,
                #     "authors": authors,
                #     "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                # }

                # # For vector store
                # docs.append(Document(page_content=page_content, metadata=metadata))

                page_content = f"""[Article {idx}]
                Title: {title}
                **Authors**: {authors}
                Publication ID: {pmid}
                Abstract: {abstract}"""
                

                metadata = {
                    "pubmed_id": pmid,
                    "title": title,
                    "authors": authors,
                    "Article Number":idx,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }

                docs.append(Document(page_content=page_content, metadata=metadata))

                # For Supabase (optional)
                articles_data.append({
                    "topic_id": topic_id,
                    "pubmed_id": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
            except Exception as parse_err:
                logger.warning(f"⚠️ Skipping malformed article: {parse_err}")

        logger.info(f"📄 Processed {len(docs)} valid articles.")

        # Step 4: Split documents
      
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,    # increase chunk size enough to hold entire article
            chunk_overlap=0,    # no overlap needed if you want clean, separate chunks
            separators=["\n\n", "\n", ".", " "]  # allow splitting at paragraphs or sentences if needed
        )

        split_docs = text_splitter.split_documents(docs)

        # Step 5: Create and save vector store
        temp = FAISS.from_documents(split_docs, embeddings)
        temp.save_local(f"vectorstores/{topic_id}")
        ids = vector_store.add_documents(documents=split_docs)
        logger.info(f"✅ Vector store created and saved for topic_id '{topic_id}'")

        # Step 6 (optional): Store metadata in Supabase
        if supabase:
            supabase.table("topics").update({
                "status": "completed"
            }).eq("id", topic_id).execute()

            supabase.table("articles").insert(articles_data).execute()
            logger.info(f"✅ Stored {len(articles_data)} articles metadata to Supabase.")

        return True

    except Exception as e:
        logger.error(f"❌ Error in fetch_and_create_vectorstore: {str(e)}")
        if supabase:
            supabase.table("topics").update({"status": f"error: {str(e)}", "article_count": 0}).eq("id", topic_id).execute()
        return False

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
            combine_docs_chain_kwargs={"prompt": detailed_prompt}
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