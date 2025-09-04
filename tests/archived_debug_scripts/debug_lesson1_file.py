"""
Debug Lesson 1 PDF Processing

Test if lesson1.pdf can be read and processed correctly.
"""

import sys
from pathlib import Path

sys.path.append('../processing')
from embed_docs import DocumentProcessor


def test_lesson1_extraction():
    """Test extracting content from lesson1.pdf"""
    
    lesson1_path = Path("../documents/foundational/lesson1.pdf")
    
    print("=" * 50)
    print("TESTING LESSON 1 PDF EXTRACTION")
    print("=" * 50)
    
    print(f"File path: {lesson1_path}")
    print(f"File exists: {lesson1_path.exists()}")
    print(f"File size: {lesson1_path.stat().st_size if lesson1_path.exists() else 'N/A'} bytes")
    
    if not lesson1_path.exists():
        print("❌ lesson1.pdf not found!")
        return
    
    # Test extraction
    try:
        processor = DocumentProcessor()
        
        print("\\n🔍 Attempting text extraction...")
        text = processor.extract_text_from_file(lesson1_path)
        
        print(f"\\nExtracted text length: {len(text)} characters")
        
        if text:
            print(f"\\nFirst 500 characters:")
            print("-" * 40)
            print(text[:500])
            print("-" * 40)
            
            # Check for lesson markers
            lesson_markers = []
            for i in range(1, 7):
                if f"lesson {i}" in text.lower() or f"=== lesson {i}" in text.lower():
                    lesson_markers.append(i)
            
            print(f"\\nFound lesson markers: {lesson_markers}")
            
            # Test processing into chunks
            print(f"\\n🔄 Testing document processing...")
            chunks = processor.process_document(
                file_path=lesson1_path,
                course_name="foundational",
                professor_name="Professor Robert Ceresa"
            )
            
            print(f"\\nProcessed into {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3
                print(f"\\nChunk {i+1}:")
                print(f"  Length: {len(chunk.text)} chars")
                print(f"  Lesson: {chunk.metadata.get('lesson_number', 'none')}")
                print(f"  Text: {chunk.text[:200]}...")
        else:
            print("❌ No text extracted from PDF!")
            
    except Exception as e:
        print(f"❌ Error processing lesson1.pdf: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_lesson1_extraction()