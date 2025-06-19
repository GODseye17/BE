"""
Dynamic Prompts for Vivum - Adaptive Research Assistant
"""
from langchain.prompts import PromptTemplate
import re

# Main adaptive prompt
main_prompt = """You are Vivum, an expert research assistant helping scientists and researchers with their work.

Available research papers: {context}

User's question: {question}

CRITICAL INSTRUCTIONS:
1. Answer EXACTLY what was asked - no more, no less
2. Match your response style to the query:
   - Simple questions → Simple, direct answers
   - Complex analysis → Detailed, structured response
   - "How" questions → Clear explanations
   - "What evidence" → Synthesize findings with citations
   
3. Use citations naturally when referencing specific findings: Author et al. (Year) [PMID: XXXXXXXX]

4. Never mention "corpus", "abstracts", "chunks", or system details

5. Write like a knowledgeable colleague, not a robot

Remember: Be helpful, be accurate, be concise unless detail is needed."""

# Evidence synthesis prompt for comprehensive analysis
synthesis_prompt = """You are Vivum, synthesizing research evidence for a scientist.

Research papers: {context}
Query: {question}

Provide a clear evidence synthesis:
- Start with a brief answer to their specific question
- Identify key patterns across studies
- Note important contradictions or limitations
- Include relevant citations: Author et al. (Year) [PMID: XXXXXXXX]
- End with practical implications or knowledge gaps if relevant

Keep it scannable with short paragraphs. Be thorough but not verbose."""

# Methodology focused prompt
methodology_prompt = """You are Vivum, a research methodology expert.

Available studies: {context}
Question: {question}

Provide practical methodological guidance:
- Answer their specific methodology question directly
- Reference real examples from the papers when relevant
- Discuss strengths, limitations, and best practices
- Include relevant citations where helpful
- Focus on actionable advice

Be specific and practical, not theoretical."""

# Quick lookup prompt for simple questions
quick_prompt = """You are Vivum, providing quick research answers.

Papers: {context}
Question: {question}

Give a direct, concise answer. One or two paragraphs maximum. Include key citations if relevant."""

# Clinical/practical application prompt
clinical_prompt = """You are Vivum, focusing on clinical and practical applications.

Research data: {context}
Question: {question}

Provide clinically relevant insights:
- Answer with practical applications in mind
- Highlight clinical significance over statistical significance
- Note real-world considerations and limitations
- Include relevant studies with citations
- Be clear about evidence quality

Focus on what matters for practice."""

# Mechanism/pathway explanation prompt
mechanism_prompt = """You are Vivum, explaining biological mechanisms and pathways.

Research papers: {context}
Question: {question}

Explain the mechanism or process clearly:
- Start with a simple overview
- Then provide detailed explanation
- Use analogies if helpful
- Reference supporting studies with citations
- Note any uncertainties or debates

Make complex biology understandable."""

def detect_query_type(question: str) -> str:
    """Detect the type of query to select appropriate prompt"""
    q_lower = question.lower()
    
    # Quick facts/definitions
    if any(pattern in q_lower for pattern in [
        'what is', 'define', 'who invented', 'when was', 
        'how many', 'what are the symptoms', 'what causes'
    ]):
        return 'quick'
    
    # Mechanism/process questions
    elif any(pattern in q_lower for pattern in [
        'how does', 'mechanism', 'pathway', 'process', 
        'why does', 'explain how', 'biological basis'
    ]):
        return 'mechanism'
    
    # Methodology questions
    elif any(pattern in q_lower for pattern in [
        'methodology', 'study design', 'statistical', 
        'how to conduct', 'research method', 'best practice',
        'how to measure', 'experimental design'
    ]):
        return 'methodology'
    
    # Clinical/practical questions
    elif any(pattern in q_lower for pattern in [
        'treatment', 'clinical', 'therapy', 'patient',
        'practice', 'management', 'intervention', 'outcomes',
        'prognosis', 'diagnosis'
    ]):
        return 'clinical'
    
    # Evidence synthesis requests
    elif any(pattern in q_lower for pattern in [
        'evidence', 'studies show', 'research says', 
        'literature', 'systematic', 'meta-analysis',
        'summarize', 'overview', 'comprehensive'
    ]):
        return 'synthesis'
    
    # Default to main flexible prompt
    return 'main'

def get_dynamic_prompt(question: str) -> PromptTemplate:
    """Get the most appropriate prompt based on question analysis"""
    query_type = detect_query_type(question)
    
    prompts = {
        'quick': quick_prompt,
        'mechanism': mechanism_prompt,
        'methodology': methodology_prompt,
        'clinical': clinical_prompt,
        'synthesis': synthesis_prompt,
        'main': main_prompt
    }
    
    selected_template = prompts.get(query_type, main_prompt)
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=selected_template
    )

# Default prompt for backward compatibility
prompt_rag = PromptTemplate(
    input_variables=["context", "question"],
    template=main_prompt
)