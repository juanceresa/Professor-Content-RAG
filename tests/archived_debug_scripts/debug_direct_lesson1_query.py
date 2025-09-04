"""
Debug Direct Lesson 1 Query

Directly query Pinecone to see if lesson 1 content exists and why search isn't finding it.
"""

import sys
import toml
from pathlib import Path

sys.path.append('..')

from pinecone import Pinecone


def debug_direct_lesson1_query():
    """Directly query for lesson 1 content"""
    
    # Initialize Pinecone
    secrets_path = Path("../.streamlit/secrets.toml")
    secrets = toml.load(secrets_path)
    
    pc = Pinecone(api_key=secrets["PINECONE_API_KEY"])
    index = pc.Index(secrets["PINECONE_INDEX_NAME"])
    
    print("=" * 60)
    print("DIRECT LESSON 1 QUERY DEBUG")
    print("=" * 60)
    
    # Query foundational-lessons namespace specifically for lesson 1
    namespace = "foundational-lessons"
    
    print(f"\\nQuerying {namespace} for lesson 1 content...")
    
    # Try different filter variations
    filter_tests = [
        {"lesson_number": "1"},
        {"lesson_number": 1},
        {"content_type": "lesson", "lesson_number": "1"},
        {"content_type": "lesson", "course": "foundational", "lesson_number": "1"},
        {}  # No filter to see all content
    ]
    
    for i, test_filter in enumerate(filter_tests):
        print(f"\\n--- Test {i+1}: Filter = {test_filter} ---")
        
        try:
            results = index.query(
                vector=[0.0] * 768,
                top_k=5,
                include_metadata=True,
                namespace=namespace,
                filter=test_filter if test_filter else None
            )
            
            print(f"Found {len(results.matches)} results:")
            
            for j, match in enumerate(results.matches):
                metadata = match.metadata
                text = metadata.get('text', '')
                
                print(f"\\n  Result {j+1}:")
                print(f"    Lesson Number: {metadata.get('lesson_number', 'NONE')}")
                print(f"    Content Type: {metadata.get('content_type', 'NONE')}")
                print(f"    Course: {metadata.get('course', 'NONE')}")
                print(f"    Source: {metadata.get('source', 'NONE')}")
                print(f"    Text (100 chars): {text[:100]}...")
                
                # Check if this is actually lesson 1 content
                is_lesson1 = ('lesson 1' in text.lower() or 
                             'understanding politics' in text.lower() or
                             metadata.get('lesson_number') == '1')
                print(f"    Is Lesson 1: {'✅' if is_lesson1 else '❌'}")
            
        except Exception as e:
            print(f"❌ Error with filter {test_filter}: {e}")
    
    # Also check what the exact search filter should be based on dual_content_handler
    print("\\n" + "=" * 60)
    print("CHECKING DUAL SEARCH FILTER")
    print("=" * 60)
    
    from dual_content_handler import DualContentHandler
    
    dual_handler = DualContentHandler(index)
    namespaces = dual_handler.get_content_namespaces("foundational", "1")
    
    print(f"\\nDual handler namespaces:")
    print(f"  Mastery: {namespaces.mastery}")
    print(f"  Lessons: {namespaces.lessons}")
    print(f"  Lesson filter: {namespaces.lesson_filter}")
    
    # Test with the exact filter that should be used
    if namespaces.lesson_filter:
        print(f"\\n--- Testing exact dual search filter ---")
        print(f"Filter: {namespaces.lesson_filter}")
        
        try:
            exact_results = index.query(
                vector=[0.0] * 768,
                top_k=10,
                include_metadata=True,
                namespace=namespaces.lessons,
                filter=namespaces.lesson_filter
            )
            
            print(f"Found {len(exact_results.matches)} results with exact filter:")
            
            for k, match in enumerate(exact_results.matches):
                metadata = match.metadata
                text = metadata.get('text', '')
                print(f"\\n  Match {k+1}:")
                print(f"    Text: {text[:150]}...")
                print(f"    Metadata: {metadata}")
                
        except Exception as e:
            print(f"❌ Error with exact filter: {e}")


if __name__ == "__main__":
    debug_direct_lesson1_query()