#!/usr/bin/env python3
"""
Debug Content Extraction

Test script to examine what pdfplumber extracts from lesson PDFs
and how the improved chunking algorithm processes it.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('processing')

from processing.embed_docs import DocumentProcessor
from improved_chunking import create_improved_chunks

def debug_pdf_extraction():
    """Debug PDF extraction and chunking for foundational lesson 1"""
    
    # Initialize document processor
    processor = DocumentProcessor()
    
    # Test files to check
    test_files = [
        Path("documents/foundational/lesson1.pdf"),
        Path("documents/foundational/learning/MASTER Group 2 A Foundational Lecture copy 10.pdf"),
    ]
    
    for file_path in test_files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        print(f"\n" + "="*80)
        print(f"🔍 ANALYZING: {file_path.name}")
        print("="*80)
        
        # Step 1: Test PDF extraction
        print(f"\n1️⃣ PDF EXTRACTION with pdfplumber:")
        extracted_text = processor.extract_text_from_pdf(file_path)
        
        print(f"   📊 Extracted {len(extracted_text)} characters")
        print(f"   📄 First 500 characters:")
        print(f"   {extracted_text[:500]}...")
        
        # Search for the politics definition
        if "politics is" in extracted_text.lower():
            print(f"   ✅ FOUND 'politics is' in extracted text!")
            # Find the context around it
            text_lower = extracted_text.lower()
            pos = text_lower.find("politics is")
            context_start = max(0, pos - 100)
            context_end = min(len(extracted_text), pos + 200)
            print(f"   📝 Context: {extracted_text[context_start:context_end]}")
        else:
            print(f"   ❌ 'politics is' NOT found in extracted text")
            
        # Search for "understanding politics"
        if "understanding politics" in extracted_text.lower():
            print(f"   ✅ FOUND 'understanding politics' in extracted text!")
            text_lower = extracted_text.lower()
            pos = text_lower.find("understanding politics")
            context_start = max(0, pos - 50)
            context_end = min(len(extracted_text), pos + 300)
            print(f"   📝 Context: {extracted_text[context_start:context_end]}")
        else:
            print(f"   ❌ 'understanding politics' NOT found in extracted text")
            
        # Search for "world building"
        if "world building" in extracted_text.lower():
            print(f"   ✅ FOUND 'world building' in extracted text!")
            text_lower = extracted_text.lower()
            pos = text_lower.find("world building")
            context_start = max(0, pos - 100)
            context_end = min(len(extracted_text), pos + 200)
            print(f"   📝 Context: {extracted_text[context_start:context_end]}")
        else:
            print(f"   ❌ 'world building' NOT found in extracted text")
        
        # Step 2: Test improved chunking
        print(f"\n2️⃣ IMPROVED CHUNKING:")
        try:
            chunks = create_improved_chunks(extracted_text, lesson_number="1")
            print(f"   📊 Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\n   Chunk {i+1} ({len(chunk)} chars):")
                print(f"   {chunk[:200]}...")
                
                # Check if this chunk contains the definition
                if "politics is" in chunk.lower():
                    print(f"   ✅ This chunk contains 'politics is'!")
                if "world building" in chunk.lower():
                    print(f"   ✅ This chunk contains 'world building'!")
                    
        except Exception as e:
            print(f"   ❌ Chunking failed: {e}")
            
        print(f"\n" + "-"*80)

if __name__ == "__main__":
    print("🔧 Content Extraction Debug Tool")
    print("This will help us understand what happened to the politics definition.")
    debug_pdf_extraction()