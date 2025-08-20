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

# OpenAI Configuration (Optional - for Critic Agent)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional

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

# Performance Configuration
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Rate Limiting
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))

# Timeouts
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# Memory Management
MAX_MEMORY_USAGE = int(os.getenv("MAX_MEMORY_USAGE", "2048"))  # MB
ENABLE_MEMORY_MONITORING = os.getenv("ENABLE_MEMORY_MONITORING", "true").lower() == "true"

# Export all settings
__all__ = [
    'SUPABASE_URL', 'SUPABASE_KEY', 'TOGETHER_API_KEY', 'OPENAI_API_KEY', 'LLM_MODEL',
    'LLM_TEMPERATURE', 'LLM_MAX_TOKENS', 'EMBEDDING_MODEL',
    'CLEANUP_INTERVAL_HOURS', 'CLEANUP_DAYS_OLD', 'MAX_CONVERSATIONS', 'PORT',
    'ENABLE_CACHING', 'REDIS_URL', 'CACHE_TTL', 'RATE_LIMIT_ENABLED', 
    'MAX_REQUESTS_PER_MINUTE', 'REQUEST_TIMEOUT', 'LLM_TIMEOUT', 
    'MAX_MEMORY_USAGE', 'ENABLE_MEMORY_MONITORING', 'logger'
]