#!/bin/bash

# GraphRAG Pipeline Test Runner
# This script sets up the environment and runs the test

echo "🚀 GraphRAG Pipeline Test Runner"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "simple_graphrag_test.py" ]; then
    echo "❌ Test script not found. Please run this from the project root directory."
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating one with template..."
    cat > .env << EOF
# GraphRAG Pipeline Environment Variables
# Required for full functionality

# Together AI API Key (Required for LLM functionality)
# Get from: https://together.ai/
TOGETHER_API_KEY=your_together_api_key_here

# Supabase Configuration (Required for database)
# Get from: https://supabase.com/
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# OpenAI API Key (Optional - for critic agent)
# Get from: https://platform.openai.com/
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration (Optional - defaults provided)
EMBEDDING_MODEL=sentence-transformers/multi-qa-mpnet-base-dot-v1
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=4096
EOF
    echo "📝 Created .env template. Please edit it with your API keys."
    echo "   Then run this script again."
    exit 1
fi

# Load environment variables
echo "🔍 Loading environment variables..."
source .env

# Check for required API keys
missing_keys=()
if [ -z "$TOGETHER_API_KEY" ] || [ "$TOGETHER_API_KEY" = "your_together_api_key_here" ]; then
    missing_keys+=("TOGETHER_API_KEY")
fi

if [ -z "$SUPABASE_URL" ] || [ "$SUPABASE_URL" = "your_supabase_url_here" ]; then
    missing_keys+=("SUPABASE_URL")
fi

if [ -z "$SUPABASE_KEY" ] || [ "$SUPABASE_KEY" = "your_supabase_key_here" ]; then
    missing_keys+=("SUPABASE_KEY")
fi

if [ ${#missing_keys[@]} -gt 0 ]; then
    echo "❌ Missing required API keys:"
    for key in "${missing_keys[@]}"; do
        echo "   - $key"
    done
    echo ""
    echo "Please edit the .env file with your actual API keys."
    echo "You can get them from:"
    echo "   - Together AI: https://together.ai/"
    echo "   - Supabase: https://supabase.com/"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import langchain, supabase, transformers" 2>/dev/null; then
    echo "⚠️  Some dependencies may be missing."
    echo "   Installing requirements..."
    pip install -r requirements.txt
fi

# Run the test
echo "🎯 Starting GraphRAG Pipeline Test..."
echo "=================================="
python3 simple_graphrag_test.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
else
    echo ""
    echo "❌ Test failed. Check the logs above for details."
    exit 1
fi
