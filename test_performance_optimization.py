"""
Test script to demonstrate performance optimizations
"""
import requests
import time
import asyncio

BASE_URL = "http://localhost:8000"

def test_performance_comparison():
    """Compare performance with and without embeddings"""
    
    test_query = "latest advances in diabetes treatment"
    
    print("🚀 Performance Optimization Test")
    print("=" * 60)
    
    # Test 1: With embeddings (default - full RAG capability)
    print("\n1️⃣ FULL RAG MODE (with embeddings):")
    print("-" * 40)
    
    start_time = time.time()
    response1 = requests.post(
        f"{BASE_URL}/fetch-topic-data",
        json={
            "topic": test_query,
            "max_results": 20,
            "create_embeddings": True  # Default, but being explicit
        }
    )
    
    if response1.status_code == 200:
        data1 = response1.json()
        topic_id_with_embeddings = data1["topic_id"]
        print(f"✅ Topic ID: {topic_id_with_embeddings}")
        
        # Wait for completion
        print("⏳ Waiting for completion...")
        while True:
            status_resp = requests.get(f"{BASE_URL}/topic/{topic_id_with_embeddings}/status")
            status = status_resp.json()["status"]
            
            if status == "completed":
                elapsed_with_embeddings = time.time() - start_time
                print(f"✅ Completed in: {elapsed_with_embeddings:.2f} seconds")
                print("✅ RAG queries enabled!")
                break
            elif status.startswith("error"):
                print(f"❌ Error: {status}")
                break
            
            time.sleep(2)
    
    # Test 2: Without embeddings (metadata only - fast mode)
    print("\n2️⃣ FAST METADATA MODE (no embeddings):")
    print("-" * 40)
    
    start_time = time.time()
    response2 = requests.post(
        f"{BASE_URL}/fetch-topic-data",
        json={
            "topic": test_query,
            "max_results": 20,
            "create_embeddings": False  # Skip embeddings for speed
        }
    )
    
    if response2.status_code == 200:
        data2 = response2.json()
        topic_id_metadata_only = data2["topic_id"]
        print(f"✅ Topic ID: {topic_id_metadata_only}")
        
        # Wait for completion
        print("⏳ Waiting for completion...")
        while True:
            status_resp = requests.get(f"{BASE_URL}/topic/{topic_id_metadata_only}/status")
            status = status_resp.json()["status"]
            
            if status == "completed":
                elapsed_metadata_only = time.time() - start_time
                print(f"✅ Completed in: {elapsed_metadata_only:.2f} seconds")
                print("⚠️  RAG queries NOT available (metadata only)")
                break
            elif status.startswith("error"):
                print(f"❌ Error: {status}")
                break
            
            time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    if 'elapsed_with_embeddings' in locals() and 'elapsed_metadata_only' in locals():
        speedup = elapsed_with_embeddings / elapsed_metadata_only
        time_saved = elapsed_with_embeddings - elapsed_metadata_only
        
        print(f"Full RAG Mode: {elapsed_with_embeddings:.2f}s")
        print(f"Metadata Only: {elapsed_metadata_only:.2f}s")
        print(f"⚡ Speedup: {speedup:.1f}x faster")
        print(f"⏱️  Time Saved: {time_saved:.2f}s")
        
        print("\n📝 Use Cases:")
        print("- Full RAG: When users need to ask questions about articles")
        print("- Metadata: When just listing/exporting article data")
    
    # Test 3: Check if we can query the RAG-enabled topic
    print("\n3️⃣ TESTING RAG CAPABILITY:")
    print("-" * 40)
    
    if 'topic_id_with_embeddings' in locals():
        query_response = requests.post(
            f"{BASE_URL}/query",
            json={
                "topic_id": topic_id_with_embeddings,
                "query": "What are the main findings about diabetes treatment?"
            }
        )
        
        if query_response.status_code == 200:
            print("✅ RAG query successful on embeddings-enabled topic")
        else:
            print(f"❌ RAG query failed: {query_response.status_code}")
    
    # Try to query the metadata-only topic (should fail)
    if 'topic_id_metadata_only' in locals():
        query_response2 = requests.post(
            f"{BASE_URL}/query",
            json={
                "topic_id": topic_id_metadata_only,
                "query": "What are the main findings?"
            }
        )
        
        if query_response2.status_code != 200:
            print("✅ Correctly rejected RAG query on metadata-only topic")
        else:
            print("⚠️  Unexpected: RAG query worked on metadata-only topic")

def test_different_batch_sizes():
    """Test performance with different numbers of articles"""
    print("\n\n🔬 BATCH SIZE PERFORMANCE TEST")
    print("=" * 60)
    
    batch_sizes = [5, 10, 20, 50]
    results = []
    
    for size in batch_sizes:
        print(f"\nTesting with {size} articles...")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/fetch-topic-data",
            json={
                "topic": "COVID-19 vaccine efficacy",
                "max_results": size,
                "create_embeddings": True
            }
        )
        
        if response.status_code == 200:
            topic_id = response.json()["topic_id"]
            
            # Wait for completion
            while True:
                status_resp = requests.get(f"{BASE_URL}/topic/{topic_id}/status")
                status = status_resp.json()["status"]
                
                if status == "completed":
                    elapsed = time.time() - start_time
                    results.append((size, elapsed))
                    print(f"✅ {size} articles: {elapsed:.2f}s ({elapsed/size:.2f}s per article)")
                    break
                elif status.startswith("error"):
                    print(f"❌ Error: {status}")
                    break
                
                time.sleep(2)
    
    # Show results
    if results:
        print("\n📊 PERFORMANCE SCALING:")
        print("-" * 40)
        print("Articles | Time (s) | Per Article")
        print("-" * 40)
        for size, elapsed in results:
            print(f"{size:8d} | {elapsed:8.2f} | {elapsed/size:6.2f}s")

if __name__ == "__main__":
    print("🏃 Running Vivum Performance Optimization Tests\n")
    
    # Run the main comparison test
    test_performance_comparison()
    
    # Optional: Test different batch sizes
    print("\n\nPress Enter to run batch size tests (or Ctrl+C to skip)...")
    try:
        input()
        test_different_batch_sizes()
    except KeyboardInterrupt:
        print("\nSkipping batch size tests.")
    
    print("\n\n✅ All tests completed!")