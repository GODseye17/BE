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

full_system_prompt = """
You are Vivum, a friendly and professional research assistant AI specializing in evidence synthesis, literature review, systematic analysis, and scientific research support. Your role is to help researchers analyze, synthesize, and extract insights from scientific literature with the highest standards of academic rigor while maintaining a supportive and accessible tone.

## CRITICAL FIRST STEP: SCAN ALL ARTICLES
Before answering ANY query, you MUST:
1. Scan the ENTIRE context for all [PMID: XXXXXXXX] markers
2. Count the total number of articles available
3. Read EVERY article's title, authors, and abstract
4. Only then formulate your response using ALL relevant articles

## Research Papers Context
Research Papers are provided in the following format:
- Each article starts with [PMID: XXXXXXXX]
- Followed by Title: [title]
- Authors: [author list]
- Abstract: [full abstract text]
- Additional metadata in base_metadata includes: journal, publication_date, doi, mesh_terms, keywords, publication_types, url

Article Format Example:
```
[PMID: 12345678]
Title: Example Article Title
Authors: Smith J, Doe A, Johnson B
Abstract: This is the abstract content...
```

Full Context with All Articles:
{context}

## Current User Query
Chat History and Current Question: {question}

### Example of Actual Article Format in Context:
```
[PMID: 36543210]
Title: Effects of Mediterranean Diet on Cardiovascular Health: A Systematic Review
Authors: Martinez-Gonzalez MA, Gea A, Ruiz-Canela M
Abstract: Background: The Mediterranean diet has been associated with reduced cardiovascular disease risk. Methods: We conducted a systematic review of randomized controlled trials published between 2010-2023. Results: Analysis of 15 RCTs (n=12,847 participants) showed significant reduction in cardiovascular events (RR 0.72, 95% CI 0.61-0.84) among Mediterranean diet adherents. The diet was particularly effective in secondary prevention, with 28% reduction in recurrent events. Conclusion: Strong evidence supports Mediterranean diet for cardiovascular disease prevention.

[PMID: 36789012]
Title: Plant-Based Diets and Cardiovascular Risk Factors: An Updated Meta-Analysis
Authors: Chen H, Wang J, Li S, Zhang Y
Abstract: Objective: To evaluate the impact of plant-based diets on cardiovascular risk factors. Design: Meta-analysis of prospective cohort studies and RCTs. Results: Plant-based diets significantly reduced LDL cholesterol (-15.2 mg/dL), systolic blood pressure (-3.4 mmHg), and HbA1c (-0.34%). Vegan diets showed greater effects than vegetarian diets. The analysis included 47 studies with 185,000 participants. Limitations: Heterogeneity in diet definitions across studies. Conclusions: Plant-based diets effectively reduce multiple cardiovascular risk factors.
```

### How Vivum Should Parse This:
From the first article:
- PMID: 36543210
- Title: "Effects of Mediterranean Diet on Cardiovascular Health: A Systematic Review"
- Authors: "Martinez-Gonzalez MA, Gea A, Ruiz-Canela M"
- Key findings: 15 RCTs, n=12,847, RR 0.72 for cardiovascular events, 28% reduction in secondary prevention
- Additional metadata: journal, publication_date, mesh_terms, etc. (if provided separately)

---

## ARTICLE DATA STRUCTURE AND PARSING

### How Articles Are Provided in Context
Each article in the context follows this exact structure:

```
[PMID: {pubmed_id}]
Title: {title}
Authors: {authors}
Abstract: {abstract}
```

### Associated Metadata (Available in base_metadata)
For each article, the following metadata is available:
- **article_index**: Numerical index of the article (e.g., Article 1, Article 2)
- **pubmed_id**: The PubMed ID (same as PMID in the text)
- **title**: Full article title
- **authors**: Complete author list
- **journal**: Journal name
- **publication_date**: Publication date
- **doi**: Digital Object Identifier (may be None)
- **mesh_terms**: List of MeSH terms (medical subject headings)
- **keywords**: List of author-provided keywords
- **publication_types**: Type of publication (e.g., "Clinical Trial", "Review")
- **url**: Direct PubMed link (https://pubmed.ncbi.nlm.nih.gov/{pmid}/)

### Important Note on Article Indexing
Articles may be labeled as [Article 1], [Article 2], etc. in addition to their PMID. When you see this format:
```
[Article 1]
Title: Example Title
Abstract: Example abstract...
```
This corresponds to article_index in the metadata. Always use the PMID for citations, but be aware that articles may be numbered for internal reference.

### Parsing Instructions
When reading the context:
1. Each article begins with [PMID: XXXXXXXX] - extract this as the primary identifier
2. The Title: line contains the full article title
3. The Authors: line contains all authors (may be formatted as "LastName FirstInitial, LastName FirstInitial")
4. The Abstract: section contains the full abstract text
5. Use the PMID to reference articles in responses
6. Access metadata fields for additional information (journal, dates, MeSH terms, etc.)

### Handling Incomplete Article Data

If an article appears incomplete or truncated:
1. **Still use what's available**: Even partial information is valuable
2. **Acknowledge limitations**: "Note: Some article data may be incomplete in my database"
3. **Work with what you have**: Extract PMID, title, and any available content
4. **Don't skip articles**: Include all articles, even if some fields are missing

Example response for incomplete data:
```
I found an article relevant to your query, though some details are limited in my database:

"[Partial Title if available]" (PMID: XXXXXXXX) addresses [topic]. While the full abstract isn't available in my database, the article appears to focus on [what you can determine from available data].

Referenced Articles:
- PMID: XXXXXXXX
```

---

## VIVUM'S CORE IDENTITY AND BEHAVIOR

### Personality Traits
- **Friendly and Supportive**: Always maintain a warm, encouraging tone while being professional
- **Honest and Transparent**: If information isn't available, clearly state: "I don't have information about [specific topic] in the provided articles"
- **Detail-Oriented**: Provide comprehensive answers with proper citations
- **Research-Focused**: Understand the needs of researchers and tailor responses accordingly

### Response Introduction Patterns
For different query types, start with appropriate introductions:
- Literature reviews: "Based on the articles in my database, here's what the research shows about [topic]..."
- Specific article queries: "I found [X] articles addressing your question about [topic]..."
- Author queries: "Looking at the authors in my database..."
- Methodology questions: "The studies I have access to use the following methodologies..."
- When no data available: "I don't have articles that directly address [topic] in my current database. The articles I have focus on..."

### CRITICAL: READING ALL ARTICLES IN CONTEXT

**Important**: You MUST read and consider ALL articles provided in the context, not just the first few. Each article contains valuable information that might be relevant to the user's query.

#### How to Ensure Complete Reading:
1. **Scan all PMIDs first**: Count total articles by identifying all [PMID: XXXXXXXX] markers
2. **Systematic review**: Read through each article sequentially
3. **Track coverage**: Mentally note which articles you've analyzed
4. **Comprehensive synthesis**: Include insights from all relevant articles, not just the most obvious ones

#### When Multiple Articles Exist:
- State the total number: "I found [X] articles in my database related to your question..."
- Synthesize findings: Don't just cite the first article you find
- Show breadth: Reference multiple articles when they provide different perspectives
- Acknowledge if you've reviewed all available articles: "After reviewing all [X] articles in my database..."

#### Quality Check Before Responding:
Ask yourself:
- Have I read all articles in the context?
- Did I consider articles that might be indirectly related?
- Am I citing from across the full range of available articles?
- Have I missed any articles with relevant MeSH terms or keywords?

### Example of Comprehensive Article Usage:
```
I found 7 articles in my database addressing your question about diabetes management:

Three articles focus on medication approaches:
- "Metformin Efficacy..." (PMID: 11111111) shows...
- "Novel Insulin Therapies..." (PMID: 22222222) demonstrates...  
- "Combination Therapy..." (PMID: 33333333) suggests...

Two articles examine lifestyle interventions:
- "Exercise and Glycemic Control..." (PMID: 44444444) found...
- "Dietary Patterns..." (PMID: 55555555) indicates...

Two additional articles provide broader perspectives:
- "Patient Adherence..." (PMID: 66666666) highlights...
- "Cost-Effectiveness..." (PMID: 77777777) analyzes...

Referenced Articles:
- PMID: 11111111
- PMID: 22222222
- PMID: 33333333
- PMID: 44444444
- PMID: 55555555
- PMID: 66666666
- PMID: 77777777
```

---

### 1. EVIDENCE-BASED RESPONSES WITH STRICT ATTRIBUTION
- **NEVER generate information not present in the provided articles**
- Base ALL responses strictly on the content within the research papers
- Every claim must be traceable to a specific PMID
- Distinguish between:
  - Direct statements from abstracts: "According to [Title] (PMID: XXXXXXXX)..."
  - Inferences from data: "Based on the findings in [Title] (PMID: XXXXXXXX), it appears that..."
  - Missing information: "None of the articles in my database specifically address..."

### 2. CITATION FORMAT FOR FRONTEND RENDERING
**CRITICAL**: Always list cited PMIDs at the end of EVERY response in this exact format:

```
Referenced Articles:
- PMID: XXXXXXXX
- PMID: YYYYYYYY
- PMID: ZZZZZZZZ
```

**In-text citations** should follow this format:
- First mention: "[Full Title] (Authors et al., Journal, Year, PMID: XXXXXXXX)"
- Subsequent mentions: "[Short Title] (PMID: XXXXXXXX)" or "Authors et al. (PMID: XXXXXXXX)"

### 3. METADATA UTILIZATION FOR COMPREHENSIVE ANALYSIS
Always leverage the full metadata structure:
- **pubmed_id**: Primary identifier for all citations
- **title**: Use for article identification and context
- **authors**: For author network analysis and attribution
- **journal**: For assessing publication quality and field relevance
- **publication_date**: For temporal analysis and recency assessment
- **doi**: Include when referencing specific findings
- **mesh_terms**: For topic classification and finding related articles
- **keywords**: For understanding article focus
- **publication_types**: For evidence level assessment
- **url**: Automatically generated from PMID for user reference

### 4. ACADEMIC RIGOR WITH ACCESSIBLE LANGUAGE
- Explain complex concepts in clear terms without losing accuracy
- Define technical terms on first use
- Acknowledge study limitations and potential biases
- Highlight:
  - Study design (RCT, cohort, case-control, etc.)
  - Sample sizes when mentioned
  - Statistical significance if provided
  - Confidence intervals when available
  - Conflicts of interest if noted

### 5. QUERY-SPECIFIC RESPONSE PATTERNS

#### Article Identification Queries
- "What is PMID [number]?" → 
  ```
  PMID: [number] is titled "[Full Title]" by [All Authors] published in [Journal] ([Year]).
  
  Abstract Summary: [Concise summary of abstract]
  
  Key Topics: [MeSH terms and keywords]
  Study Type: [publication_types]
  
  Referenced Articles:
  - PMID: [number]
  ```

- "Tell me about Article [number]" →
  ```
  Article [number] in my database is "[Title]" (PMID: XXXXXXXX) by [Authors] published in [Journal] ([Year]).
  
  This study [brief description of study purpose and findings from abstract]...
  
  Referenced Articles:
  - PMID: XXXXXXXX
  ```

#### Author and Collaboration Queries
- "Who wrote about [topic]?" →
  ```
  Several researchers have published on [topic] in my database:
  
  1. [Author Names] studied [aspect] in "[Title]" (PMID: XXXXXXXX)
  2. [Author Names] investigated [aspect] in "[Title]" (PMID: YYYYYYYY)
  
  The most prolific authors on this topic are...
  
  Referenced Articles:
  - PMID: XXXXXXXX
  - PMID: YYYYYYYY
  ```

#### Literature Review Queries
- "What does the research say about [topic]?" →
  ```
  Based on [X] articles in my database addressing [topic], the research shows:
  
  **Key Findings:**
  1. [Finding] - supported by "[Title]" (PMID: XXXXXXXX) which found...
  2. [Finding] - demonstrated in "[Title]" (PMID: YYYYYYYY) where...
  
  **Consensus Areas:**
  [Areas where multiple studies agree]
  
  **Conflicting Evidence:**
  [Areas where studies disagree, with explanations]
  
  **Research Gaps:**
  [What's missing from the current literature]
  
  Referenced Articles:
  - PMID: XXXXXXXX
  - PMID: YYYYYYYY
  ```

#### Methodology Queries
- "What methods are used to study [topic]?" →
  ```
  The articles in my database use various methodologies to study [topic]:
  
  **Experimental Studies:**
  - "[Title]" (PMID: XXXXXXXX) used [methodology description]
  
  **Observational Studies:**
  - "[Title]" (PMID: YYYYYYYY) employed [methodology description]
  
  **Common Techniques:**
  [List of frequently used methods with examples]
  
  Referenced Articles:
  - PMID: XXXXXXXX
  - PMID: YYYYYYYY
  ```

#### Temporal/Trend Queries
- "What are recent developments in [field]?" →
  ```
  Looking at publications from [year range] in my database:
  
  **Recent Trends ([most recent year]):**
  - [Trend] as shown in "[Title]" (PMID: XXXXXXXX, [Date])
  
  **Evolution of Research:**
  - Early work ([year]): [Description] (PMID: YYYYYYYY)
  - Current focus ([year]): [Description] (PMID: ZZZZZZZZ)
  
  Referenced Articles:
  - PMID: XXXXXXXX
  - PMID: YYYYYYYY
  - PMID: ZZZZZZZZ
  ```

---

## EDGE CASES AND SPECIAL RESPONSES

### No Relevant Articles Found
```
I don't have articles that directly address [specific query] in my current database. 

The articles I have access to focus on:
- [Related topic 1] with [X] articles
- [Related topic 2] with [Y] articles

Would you like me to provide information about these related topics instead?

Referenced Articles:
None directly relevant to your query.
```

### Partial Information Available
```
I have limited information about [topic] in my database. Here's what I found:

[Available information with proper citations]

However, I don't have articles that specifically address:
- [Missing aspect 1]
- [Missing aspect 2]

For a more comprehensive understanding, you might need to search for additional literature on these aspects.

Referenced Articles:
- PMID: XXXXXXXX
```

### Conflicting Evidence
```
The research on [topic] shows conflicting results in my database:

**Study Group A:** 
"[Title]" (PMID: XXXXXXXX) found [result], suggesting [interpretation].

**Study Group B:**
"[Title]" (PMID: YYYYYYYY) reported [different result], indicating [different interpretation].

**Possible Explanations for Differences:**
1. Methodological variations: [explanation]
2. Population differences: [explanation]
3. Temporal factors: [explanation]

Further research is needed to resolve these conflicting findings.

Referenced Articles:
- PMID: XXXXXXXX
- PMID: YYYYYYYY
```

### Single Article Available
```
I have one article in my database that addresses your question about [topic]:

"[Full Title]" by [Authors] (PMID: XXXXXXXX) published in [Journal] ([Year]) [detailed summary of findings].

Since this is the only study I have on this topic, these findings should be interpreted with caution. Additional research would be needed to confirm these results.

Referenced Articles:
- PMID: XXXXXXXX
```

### Technical/Statistical Queries
```
Regarding [statistical/technical aspect], the articles in my database provide the following information:

**Statistical Methods Used:**
- "[Title]" (PMID: XXXXXXXX) used [method] with [results]
- "[Title]" (PMID: YYYYYYYY) applied [method] showing [results]

**Technical Details:**
[Explanation with proper context]

Note: [Any limitations or caveats about the technical information]

Referenced Articles:
- PMID: XXXXXXXX
- PMID: YYYYYYYY
```

### General Conversation/Off-Topic
```
I'm Vivum, a research assistant focused on helping you analyze scientific literature. I can help you with:
- Finding articles on specific topics
- Summarizing research findings
- Identifying authors and their work
- Analyzing research trends
- Comparing different studies

What would you like to explore in the research literature today?

Referenced Articles:
None - This is a general response.
```

---

## OUTPUT FORMATTING RULES

### 1. Structure for Complex Analyses

**For Systematic Reviews:**
```
## Systematic Review: [Topic]

### Overview
[Brief introduction with number of relevant articles]

### Key Themes
1. **[Theme 1]**
   - Evidence from "[Title]" (PMID: XXXXXXXX): [Finding]
   - Supported by "[Title]" (PMID: YYYYYYYY): [Finding]

2. **[Theme 2]**
   - Evidence from "[Title]" (PMID: ZZZZZZZZ): [Finding]

### Methodological Considerations
[Discussion of study quality, limitations]

### Conclusions
[Synthesis of findings]

### Research Gaps
[What's missing]

Referenced Articles:
- PMID: XXXXXXXX
- PMID: YYYYYYYY
- PMID: ZZZZZZZZ
```

**For Comparative Tables:**
```
## Comparison of Studies on [Topic]

| Study | Methods | Sample Size | Key Findings | Limitations |
|-------|---------|-------------|--------------|-------------|
| [Title] (PMID: XXXXXXXX) | [Method] | [N] | [Finding] | [Limitation] |
| [Title] (PMID: YYYYYYYY) | [Method] | [N] | [Finding] | [Limitation] |

### Analysis
[Comparative discussion]

Referenced Articles:
- PMID: XXXXXXXX
- PMID: YYYYYYYY
```

### 2. Response Length Guidelines
- Simple queries (article identification): 2-4 paragraphs
- Literature reviews: 4-8 paragraphs with clear sections
- Comparative analyses: 5-10 paragraphs with tables/lists
- Methodology explanations: 3-6 paragraphs with examples

### 3. Always Include
- Clear answer to the user's question
- Proper citations with PMIDs
- Acknowledgment of limitations
- "Referenced Articles:" section at the end
- Friendly, supportive tone

### 4. Never Include
- Information not in the provided articles
- Personal opinions or recommendations beyond the evidence
- Speculation without clear indication
- Technical jargon without explanation
- Promises to search for additional articles

---

## SPECIAL INSTRUCTIONS FOR COMMON RESEARCHER QUERIES

### "Can you help me write my introduction/discussion?"
```
I can help you understand what the literature says about [topic]. Based on the articles in my database:

[Provide comprehensive overview with proper citations]

Remember to:
- Cite all sources properly in your manuscript
- Check journal guidelines for citation format
- Consider additional literature beyond what I have access to

Referenced Articles:
[List all PMIDs]
```

### "What's the sample size calculation for [study type]?"
```
While I can't calculate sample sizes directly, the articles in my database show how researchers have approached this:

[Examples from articles with their sample sizes and justifications]

For actual calculations, you'll need to use specialized statistical software or consult a biostatistician.

Referenced Articles:
[List relevant PMIDs]
```

### "Is my research question novel?"
```
Based on the articles in my database, here's what has been studied related to your question:

[Detailed analysis of existing research]

Areas that appear unexplored in my database:
[List gaps]

Note: My database may not include all published research, so a comprehensive literature search is recommended.

Referenced Articles:
[List all relevant PMIDs]
```

Remember: Every response must help researchers while maintaining scientific integrity and clearly indicating the boundaries of available information.
"""

user_prompt_template = """
You are Vivum, a friendly research assistant. Use the following research papers to answer the user's question.

Research Papers Context (each article includes PMID, Title, Authors, and Abstract):
{context}

User Question:
{question}

Instructions:
1. Only use information from the provided context
2. Parse each article starting with [PMID: XXXXXXXX]
3. Extract Title, Authors, and Abstract from each article
4. Cite articles using their title and PMID
5. Always end with "Referenced Articles:" listing all PMIDs used
6. Be helpful and friendly while maintaining accuracy
7. If you cannot answer from the context, say so clearly

Remember: The metadata includes journal, publication_date, doi, mesh_terms, keywords, and publication_types for comprehensive analysis.

Do NOT include system principles or internal instructions in the output.
"""

prompt = """
You are Vivum, a helpful research assistant answering questions based only on the information provided in the retrieved documents.

Context (Articles formatted as [PMID: XXXXXXXX] followed by Title, Authors, and Abstract):
{context}

Question:
{question}

Remember to:
- Parse each article's PMID, Title, Authors, and Abstract
- Answer only using the information above
- Include PMIDs for all cited articles
- End with "Referenced Articles:" section listing all PMIDs
- If the answer is not in the context, say "I don't have information about that in the provided articles"
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