"""
Debug Lesson 1 After Reprocessing

Check if the reprocessing fixed the lesson 1 content issues.
"""

import sys
import toml
from pathlib import Path

sys.path.append('..')

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from content_search import ContentSearchEngine
from dual_content_handler import DualContentHandler


def check_reprocessed_content():
    """Check the reprocessed content quality"""
    
    # Initialize services
    secrets_path = Path("../.streamlit/secrets.toml")
    secrets = toml.load(secrets_path)
    
    pc = Pinecone(api_key=secrets["PINECONE_API_KEY"])
    index = pc.Index(secrets["PINECONE_INDEX_NAME"])
    
    print("=" * 60)
    print("CHECKING REPROCESSED CONTENT")
    print("=" * 60)
    
    # Check namespace stats
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {})
    
    print(f"\\nTotal namespaces: {len(namespaces)}")
    print(f"Total vectors across all namespaces: {sum(ns.get('vector_count', 0) for ns in namespaces.values())}")
    
    for ns_name, ns_info in namespaces.items():
        print(f"  - {ns_name}: {ns_info.get('vector_count', 0)} vectors")
    
    # Focus on foundational course
    foundational_namespaces = [ns for ns in namespaces.keys() if 'foundational' in ns]
    
    print(f"\\nFoundational course namespaces: {foundational_namespaces}")
    
    # Test lesson 1 search
    if foundational_namespaces:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        content_search = ContentSearchEngine(embedding_model, index)
        
        print("\\n" + "=" * 40)
        print("TESTING LESSON 1 SEARCH")
        print("=" * 40)
        
        # Test the exact query that's failing
        results = content_search.search_course_content_dual(
            query="walk through lesson 1",
            course_namespace="foundational",
            selected_lesson="1",
            top_k=5
        )
        
        print(f"\\nSearch results:")
        print(f"Strategy: {results.get('search_strategy', 'unknown')}")
        print(f"Total found: {results.get('total_found', 0)}")
        
        chunks = results.get('chunks', [])
        if chunks:
            print(f"\\nFound {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks):
                text = chunk.get('text', '')
                print(f"\\nChunk {i+1}:")
                print(f"  Type: {chunk.get('content_type', 'unknown')}")
                print(f"  Lesson: {chunk.get('lesson', 'none')}")
                print(f"  Score: {chunk.get('score', 0):.4f}")
                print(f"  Length: {len(text)} chars")
                print(f"  Preview: {text[:200]}...")
        else:
            print("❌ No chunks returned!")
            
            # Try direct namespace query
            print("\\n🔍 Trying direct namespace query...")
            
            for ns in foundational_namespaces:
                print(f"\\nQuerying {ns} directly:")
                try:
                    direct_results = index.query(
                        vector=[0.0] * 768,
                        top_k=5,
                        include_metadata=True,
                        namespace=ns
                    )
                    
                    if direct_results.matches:
                        print(f"Found {len(direct_results.matches)} vectors in {ns}:")
                        for i, match in enumerate(direct_results.matches):
                            metadata = match.metadata
                            text = metadata.get('text', '')
                            print(f"  {i+1}. Lesson: {metadata.get('lesson_number', 'none')}, "
                                  f"Type: {metadata.get('content_type', 'unknown')}, "
                                  f"Text: {text[:100]}...")
                    else:
                        print(f"No vectors found in {ns}")
                        
                except Exception as e:
                    print(f"Error querying {ns}: {e}")


if __name__ == "__main__":
    check_reprocessed_content()