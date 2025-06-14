"""
Test script for the query transformation feature
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Test queries
test_queries = [
    "What is the effect of metformin on diabetes?",
    "breast cancer vs lung cancer survival rates",
    "How does aspirin work for heart attack prevention?",
    "Latest COVID-19 vaccine side effects",
    "Treatment options for rheumatoid arthritis",
    "Diagnosis of alzheimer's disease",
    "COPD treatment in elderly patients",
    "I'm researching the relationship between gut microbiome and autism",
    "What are the risk factors for stroke?",
    "Clinical trials for melanoma immunotherapy"
]

print("Testing Query Transformation Feature\n" + "="*50 + "\n")

# Test 1: Transform queries
print("1. Testing Query Transformations:")
print("-" * 50)

for query in test_queries:
    response = requests.post(
        f"{BASE_URL}/transform-query",
        json={"query": query}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Original: {data['original_query']}")
        print(f"🔄 Transformed: {data['transformed_query']}")
        if data['is_transformed']:
            print(f"📝 Explanation:\n{data['explanation']}")
    else:
        print(f"\n❌ Error for query: {query}")
        print(f"   Status: {response.status_code}")

print("\n" + "="*50 + "\n")

# Test 2: Test with actual PubMed search
print("2. Testing Full Pipeline (Transform + Fetch):")
print("-" * 50)

test_query = "What are the latest treatments for type 2 diabetes?"
print(f"\nQuery: {test_query}")

# First, transform the query
transform_response = requests.post(
    f"{BASE_URL}/transform-query",
    json={"query": test_query}
)

if transform_response.status_code == 200:
    transform_data = transform_response.json()
    print(f"Transformed to: {transform_data['transformed_query']}")
    
    # Now fetch articles with auto-transform enabled (default)
    fetch_response = requests.post(
        f"{BASE_URL}/fetch-topic-data",
        json={
            "topic": test_query,
            "max_results": 5,
            "auto_transform": True  # This is default, but being explicit
        }
    )
    
    if fetch_response.status_code == 200:
        fetch_data = fetch_response.json()
        topic_id = fetch_data["topic_id"]
        print(f"\n✅ Topic ID: {topic_id}")
        print(f"📊 Status: {fetch_data['status']}")
        
        # Wait for completion
        print("\nWaiting for data fetch to complete...")
        for i in range(30):  # Wait up to 30 seconds
            status_response = requests.get(f"{BASE_URL}/topic/{topic_id}/status")
            status = status_response.json()["status"]
            
            if status == "completed":
                print(f"✅ Fetch completed!")
                
                # Get articles to verify
                articles_response = requests.get(f"{BASE_URL}/topic/{topic_id}/articles?limit=5")
                if articles_response.status_code == 200:
                    articles_data = articles_response.json()
                    print(f"\n📚 Found {len(articles_data['articles'])} articles:")
                    for idx, article in enumerate(articles_data['articles'], 1):
                        print(f"   {idx}. {article['title'][:80]}...")
                break
            elif status.startswith("error"):
                print(f"❌ Error: {status}")
                break
            else:
                print(f"   Status: {status} (waiting...)")
                time.sleep(2)
    else:
        print(f"❌ Fetch error: {fetch_response.status_code}")

print("\n" + "="*50 + "\n")

# Test 3: Test with PubMed syntax (should not transform)
print("3. Testing with PubMed Syntax (No Transform Expected):")
print("-" * 50)

pubmed_query = "(diabetes[MeSH]) AND (metformin[Title/Abstract]) AND Clinical Trial[ptyp]"
response = requests.post(
    f"{BASE_URL}/transform-query",
    json={"query": pubmed_query}
)

if response.status_code == 200:
    data = response.json()
    print(f"\nOriginal: {data['original_query']}")
    print(f"Transformed: {data['transformed_query']}")
    print(f"Was transformed: {data['is_transformed']}")
    print(f"Explanation: {data['explanation']}")

print("\n" + "="*50)
print("\n✅ Query transformation tests completed!")