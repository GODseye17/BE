"""
Configuration settings for Vivum RAG Backend
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    raise ValueError("Missing Supabase credentials")

# LLM Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY must be set in environment variables")

LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Cleanup Configuration
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
CLEANUP_DAYS_OLD = int(os.getenv("CLEANUP_DAYS_OLD", "7"))

# Cache Configuration
MAX_CONVERSATIONS = 100

# Server Configuration
PORT = int(os.environ.get("PORT", 8000))

# Export all settings
__all__ = [
    'SUPABASE_URL', 'SUPABASE_KEY', 'TOGETHER_API_KEY', 'LLM_MODEL',
    'LLM_TEMPERATURE', 'LLM_MAX_TOKENS', 'EMBEDDING_MODEL',
    'CLEANUP_INTERVAL_HOURS', 'CLEANUP_DAYS_OLD', 'MAX_CONVERSATIONS', 'PORT',
    'logger'
]