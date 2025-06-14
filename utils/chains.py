"""
Conversation chain management utilities
"""
import logging
import re
from typing import Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from core.globals import get_globals
from vectorstore.manager import get_vectorstore_retriever
from utils.prompts import prompt_rag
from config.settings import MAX_CONVERSATIONS

logger = logging.getLogger(__name__)

def check_topic_fetch_status(topic_id: str) -> str:
    """Check if data fetching is complete for a topic"""
    globals_dict = get_globals()
    background_tasks_status = globals_dict['background_tasks_status']
    supabase = globals_dict['supabase']
    
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

def get_or_create_chain(topic_id: str, conversation_id: str, query: str):
    """Get or create a conversation chain for this topic and conversation"""
    globals_dict = get_globals()
    conversation_chains = globals_dict['conversation_chains']
    llm = globals_dict['llm']
    
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
    retriever = get_vectorstore_retriever(topic_id, query)
    
    # Create the chain
    logger.info(f"Chain components: LLM type: {type(llm).__name__}, " 
                   f"Retriever type: {type(retriever).__name__}")
   
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
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
        logger.info(f"🗑️ Cleaned up {len(chains_to_remove)} old conversation chains (kept {MAX_CONVERSATIONS} most recent)")

    return qa_chain

def validate_comprehensive_response(query: str, answer: str, topic_id: str) -> str:
    """Validate comprehensive responses and add coverage notes"""
    comprehensive_keywords = ["all articles", "each article", "create a table", "every article", "fetched"]
    if any(keyword in query.lower() for keyword in comprehensive_keywords):
        globals_dict = get_globals()
        supabase = globals_dict['supabase']
        
        # Count PMIDs in response
        pmids_in_response = set(re.findall(r'PMID: (\d+)', answer))
        
        # Get expected article count
        if supabase:
            try:
                articles_result = supabase.table("articles").select("pubmed_id").eq("topic_id", topic_id).execute()
                expected_count = len(articles_result.data)
                
                # Log coverage statistics
                coverage = (len(pmids_in_response) / expected_count * 100) if expected_count > 0 else 0
                logger.info(f"📊 Comprehensive query coverage: {len(pmids_in_response)}/{expected_count} articles ({coverage:.1f}%)")
                
                # Add note if significant discrepancy
                if len(pmids_in_response) < expected_count * 0.5:  # Less than 50%
                    logger.warning(f"⚠️ Low coverage for comprehensive query: only {coverage:.1f}%")
                    answer += f"\n\n📝 **Note**: This response includes {len(pmids_in_response)} out of {expected_count} available articles. For a complete listing of all articles, you may want to request the information in smaller, more specific queries."
                
            except Exception as e:
                logger.error(f"Error validating comprehensive response: {e}")
    
    return answer