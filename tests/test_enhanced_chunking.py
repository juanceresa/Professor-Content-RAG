#!/usr/bin/env python3
"""
Test script for enhanced hierarchical chunking system
Tests document classification, lesson detection, and hierarchical preservation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from embed_docs import DocumentProcessor

def test_document_classification():
    """Test the enhanced document classification"""
    print("🔍 Testing Document Classification\n")
    
    processor = DocumentProcessor()
    
    test_files = [
        "MASTER Group 2 F American Lecture.pdf",
        "MASTER African Read 062325.pdf", 
        "01 Preface.docx",
        "02 Chapter 2.docx",
        "Some Other Document.pdf"
    ]
    
    for filename in test_files:
        file_path = Path(filename)
        doc_type = processor.infer_document_type(file_path)
        print(f"📄 {filename}")
        print(f"   Type: {doc_type}")
        print()

def test_lesson_extraction():
    """Test lesson extraction from sample MASTER lecture content"""
    print("📚 Testing Lesson Extraction\n")
    
    processor = DocumentProcessor()
    
    # Sample MASTER lecture content
    sample_content = """
Introduction to Political Science

Lesson 1
• What is politics?
  ○ Politics involves power relationships
  ○ Politics shapes our daily lives
    ■ Government policies affect everything
    ■ From taxes to education to healthcare

Lesson 2
• The role of institutions
  ○ Formal institutions like government
  ○ Informal institutions like norms
    ■ Constitutional frameworks
    ■ Social expectations and traditions

Lesson 3
• The cultural dynamics of politics
  ○ This class takes an approach to politics that focuses on culture
    ■ Culture has its own dynamics, the dynamics of culture
    ■ The dynamics of culture have implications for politics
  ○ Politics involves the contest of ideas between groups
    ■ Values, beliefs, identities, and interests that groups embrace
"""
    
    # Test lesson structure extraction
    lessons = processor.extract_lesson_structure(sample_content, "master_lecture")
    print(f"Found {len(lessons)} lessons:")
    for lesson_num, lesson_info in lessons.items():
        print(f"  Lesson {lesson_num}: {lesson_info['title']}")
    
    print()

def test_hierarchical_structure():
    """Test hierarchical structure extraction"""
    print("🌳 Testing Hierarchical Structure Extraction\n")
    
    processor = DocumentProcessor()
    
    # Test with Lesson 3 example
    lesson3_content = """
Lesson 3
• The cultural dynamics of politics
  ○ This class takes an approach to politics that focuses on culture (as opposed to biology,
lets says), political theory, in particular.
    ■ Culture has its own dynamics, the dynamics of culture. The dynamics of culture have
implications for politics. The cultural dynamics of politics is something to consider.
  ○ Politics (with a focus on culture) involves the contest of ideas between groups,
the values, beliefs, identities, and interests that groups embrace and want others to
have.
  ○ Socioeconomic class groups are probably the two most important groups
historically.
    ■ Rich or poor, basically, together with the traditions of political theory they (rich
and poor) have developed or that have been developed on their behalf (e.g., liberal
political theory and republican).
      • The balance between liberal political theory and republican political theory
(e.g., between the rich and the middle class, and the poor) in democracy, in
the design of American government, is a good example of the cultural
dynamics of politics.
"""
    
    structured_content = processor.extract_hierarchical_structure(lesson3_content, "master_lecture")
    
    print(f"Extracted {len(structured_content)} structured items:")
    for item in structured_content[:10]:  # Show first 10 items
        level = item.get('level', 0)
        symbol = item.get('symbol', '')
        content = item.get('content', '')[:60] + "..." if len(item.get('content', '')) > 60 else item.get('content', '')
        lesson = item.get('lesson', 'No lesson')
        indent = "  " * level
        print(f"{indent}[L{level}] {symbol} {content}")
        print(f"{indent}    Lesson: {lesson}")
        print(f"{indent}    Path: {item.get('hierarchy_path', 'No path')}")
        print()

def test_chunking():
    """Test the enhanced chunking algorithm"""
    print("✂️ Testing Enhanced Chunking\n")
    
    processor = DocumentProcessor()
    
    # Sample content for chunking
    sample_content = """
Lesson 3
• The cultural dynamics of politics
  ○ This class takes an approach to politics that focuses on culture (as opposed to biology, let's say), political theory, in particular.
    ■ Culture has its own dynamics, the dynamics of culture. The dynamics of culture have implications for politics. The cultural dynamics of politics is something to consider.
  ○ Politics (with a focus on culture) involves the contest of ideas between groups, the values, beliefs, identities, and interests that groups embrace and want others to have.
  ○ Socioeconomic class groups are probably the two most important groups historically.
    ■ Rich or poor, basically, together with the traditions of political theory they (rich and poor) have developed or that have been developed on their behalf (e.g., liberal political theory and republican).
      • The balance between liberal political theory and republican political theory (e.g., between the rich and the middle class, and the poor) in democracy, in the design of American government, is a good example of the cultural dynamics of politics. Government is an institution. Institutions reflect the values, beliefs, identities, and interests of the people who create them.
      • The balance is not the only example, but it is a good one.
      • A balance of power that protects the interests of the rich and the poor is perhaps the most important design principle at the heart of American government.

Lesson 4
• Republican political theory
  ○ Limited government itself
    ■ Diagram of yin and yang
  ○ The states in federalism
    ■ Federalism is a principle of constitutional design in which the powers of government, derived from the people, are divided between a strong national government and individual states, with national power being supreme.
    ■ Federalism represents a commitment to decentralized power in the states (e.g., strong states, if you will) as well as centralized power in the national government (a strong national government).
"""
    
    # Test chunking with different document types
    print("Testing MASTER lecture chunking:")
    chunks = processor.intelligent_chunking(sample_content, "master_lecture", max_chunk_size=600)
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} (Length: {len(chunk)}) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"\nTotal chunks created: {len(chunks)}")

if __name__ == "__main__":
    print("🎓 Enhanced Hierarchical Chunking Test Suite\n")
    print("=" * 60)
    
    try:
        test_document_classification()
        print("=" * 60)
        test_lesson_extraction()
        print("=" * 60)
        test_hierarchical_structure()
        print("=" * 60)
        test_chunking()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()