"""
Debug Lesson 1 Search

Specifically tests what happens when "Lesson 1" is selected and traces
exactly what content is being retrieved and why it's incoherent.
"""

import sys
import toml
from pathlib import Path

sys.path.append('..')

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from content_search import ContentSearchEngine
from dual_content_handler import DualContentHandler


def debug_lesson1_query():
    """Debug exactly what happens with lesson 1 queries"""
    
    # Initialize services
    secrets_path = Path("../.streamlit/secrets.toml")
    secrets = toml.load(secrets_path)
    
    pc = Pinecone(api_key=secrets["PINECONE_API_KEY"])
    index = pc.Index(secrets["PINECONE_INDEX_NAME"])
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    content_search = ContentSearchEngine(embedding_model, index)
    dual_handler = DualContentHandler(index)
    
    # Test the exact scenario: foundational course, lesson 1 selected
    course_name = "foundational"
    lesson = "1"
    
    print("="*60)
    print("DEBUGGING LESSON 1 SEARCH ISSUE")
    print("="*60)
    
    # 1. Check what namespaces are being used
    namespaces = dual_handler.get_content_namespaces(course_name, lesson)
    print(f"\\nNamespaces being searched:")
    print(f"  Mastery: {namespaces.mastery}")
    print(f"  Lessons: {namespaces.lessons}")
    print(f"  Lesson filter: {namespaces.lesson_filter}")
    
    # 2. Test the actual search that's happening
    test_queries = [
        "walk through lesson 1",
        "what is lesson 1 about", 
        "explain lesson 1"
    ]
    
    for query in test_queries:
        print(f"\\n{'='*40}")
        print(f"TESTING QUERY: '{query}'")
        print(f"{'='*40}")
        
        # Get the dual search results
        results = content_search.search_course_content_dual(
            query=query,
            course_namespace=course_name,
            selected_lesson=lesson,
            top_k=5
        )
        
        print(f"\\nSearch Strategy: {results.get('search_strategy')}")
        print(f"Total Found: {results.get('total_found', 0)}")
        print(f"Mastery Found: {results.get('mastery_found', 0)}")
        print(f"Lesson Found: {results.get('lesson_found', 0)}")
        
        chunks = results.get('chunks', [])
        print(f"\\nRetrieved {len(chunks)} chunks:")
        
        for i, chunk in enumerate(chunks):
            print(f"\\n--- CHUNK {i+1} ---")
            print(f"Content Type: {chunk.get('content_type', 'unknown')}")
            print(f"Lesson: {chunk.get('lesson', 'none')}")
            print(f"Score: {chunk.get('score', 0):.4f}")
            print(f"Source: {chunk.get('source', 'unknown')}")
            print(f"Weight: {chunk.get('weight', 'none')}")
            
            text = chunk.get('text', '')
            print(f"Text Length: {len(text)} chars")
            print(f"Text Preview: {text[:200]}{'...' if len(text) > 200 else ''}")
            
            # Check if this chunk contains meaningful lesson 1 content
            has_lesson1_content = any(phrase in text.lower() for phrase in [
                'lesson 1', 'understanding politics', 'defining politics', 'world building'
            ])
            print(f"Contains Lesson 1 content: {'✅' if has_lesson1_content else '❌'}")
            
            # Check if it inappropriately contains other lessons
            other_lessons = []
            for lesson_num in ['2', '3', '4', '5', '6']:
                if f'lesson {lesson_num}' in text.lower():
                    other_lessons.append(lesson_num)
            if other_lessons:
                print(f"❌ PROBLEM: Contains other lessons: {other_lessons}")
    
    # 3. Directly query the lesson namespace to see what lesson 1 content exists
    print(f"\\n{'='*60}")
    print("DIRECT LESSON 1 NAMESPACE QUERY")
    print(f"{'='*60}")
    
    try:
        # Direct query to lessons namespace for lesson 1 content
        lesson_results = index.query(
            vector=[0.0] * 768,  # Zero vector for sampling
            top_k=10,
            include_metadata=True,
            namespace=f"{course_name}-lessons",
            filter={"content_type": "lesson", "lesson_number": "1"}
        )
        
        print(f"\\nFound {len(lesson_results.matches)} lesson 1 chunks in lessons namespace:")
        
        for i, match in enumerate(lesson_results.matches):
            metadata = match.metadata
            text = metadata.get('text', '')
            print(f"\\n--- LESSON 1 CHUNK {i+1} ---")
            print(f"Lesson Number: {metadata.get('lesson_number')}")
            print(f"Content Type: {metadata.get('content_type')}")
            print(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")
        
    except Exception as e:
        print(f"Error querying lesson namespace: {e}")
    
    # 4. Check mastery namespace for lesson 1 content
    print(f"\\n{'='*60}")
    print("MASTERY NAMESPACE LESSON 1 CONTENT")
    print(f"{'='*60}")
    
    try:
        mastery_results = index.query(
            vector=[0.0] * 768,
            top_k=10,
            include_metadata=True,
            namespace=f"{course_name}-mastery",
            filter={"content_type": "mastery", "lesson_number": "1"}
        )
        
        print(f"\\nFound {len(mastery_results.matches)} lesson 1 chunks in mastery namespace:")
        
        for i, match in enumerate(mastery_results.matches):
            metadata = match.metadata
            text = metadata.get('text', '')
            print(f"\\n--- MASTERY LESSON 1 CHUNK {i+1} ---")
            print(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")
            
    except Exception as e:
        print(f"Error querying mastery namespace: {e}")


if __name__ == "__main__":
    debug_lesson1_query()
    
    print("\\n" + "="*60)
    print("DEBUG COMPLETE")
    print("Check above for issues with lesson boundary enforcement and content quality")