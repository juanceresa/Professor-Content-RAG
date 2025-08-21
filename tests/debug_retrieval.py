#!/usr/bin/env python3
"""
Debug retrieval issues - check if search is working
"""

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

def test_retrieval():
    print("🔍 DEBUGGING RETRIEVAL SYSTEM")
    print("=" * 50)
    
    # Initialize connections
    PINECONE_API_KEY = "pcsk_UySHG_ErRr5FNDgTKZeC1ZwJSFnjBm8Ggt5aTNZEcJtpuVyYL5ST4No7J9xbWqjVo4UfN"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("ai-professor-platform")
    
    # Initialize embedding model (same as processing)
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Check namespaces
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {})
    
    print(f"📊 Found {len(namespaces)} namespaces:")
    for namespace, info in namespaces.items():
        print(f"   {namespace}: {info.get('vector_count', 0)} vectors")
    
    # Test with theory namespace (should have most content)
    test_namespace = "theory"
    if test_namespace not in namespaces:
        print(f"❌ Namespace '{test_namespace}' not found!")
        return
    
    print(f"\n🧪 Testing retrieval in '{test_namespace}' namespace")
    print(f"   Vectors available: {namespaces[test_namespace].get('vector_count', 0)}")
    
    # Test queries
    test_queries = [
        "What is Aristotle's view on politics?",
        "Explain Plato's Republic",
        "What is democracy?",
        "political theory"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Search without filter first
        print("   Testing WITHOUT filter...")
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace=test_namespace
        )
        
        print(f"   Found {len(results.matches)} results (no filter)")
        for i, match in enumerate(results.matches):
            print(f"     {i+1}. Score: {match.score:.3f}")
            print(f"        Text: {match.metadata.get('text', 'No text')[:100]}...")
            print(f"        Course: {match.metadata.get('course', 'No course')}")
        
        # Search WITH filter (like the app does)
        print("   Testing WITH course filter...")
        try:
            results_filtered = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                namespace=test_namespace,
                filter={"course": test_namespace}
            )
            
            print(f"   Found {len(results_filtered.matches)} results (with filter)")
            for i, match in enumerate(results_filtered.matches):
                print(f"     {i+1}. Score: {match.score:.3f}")
                print(f"        Text: {match.metadata.get('text', 'No text')[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Filter search failed: {e}")
    
    # Check what's actually in the metadata
    print(f"\n📋 Sample metadata from {test_namespace}:")
    try:
        sample_results = index.query(
            vector=[0.0] * 768,  # Dummy vector
            top_k=3,
            include_metadata=True,
            namespace=test_namespace
        )
        
        for i, match in enumerate(sample_results.matches):
            print(f"   Sample {i+1}:")
            metadata = match.metadata
            for key, value in metadata.items():
                if key != 'text':  # Skip long text field
                    print(f"     {key}: {value}")
            print(f"     text_length: {len(metadata.get('text', ''))}")
            
    except Exception as e:
        print(f"   ❌ Could not fetch sample metadata: {e}")

if __name__ == "__main__":
    test_retrieval()