"""
Test script for hierarchical chunking of academic content
"""

from embed_docs import DocumentProcessor, scan_data_directory, process_course_content
from pathlib import Path
import json
import sys

def test_data_directory_scan(data_dir: str = "data"):
    """Test scanning your actual data directory"""
    print("Testing Data Directory Scan...")
    print("=" * 60)

    scan_results = scan_data_directory(Path(data_dir))

    if "error" in scan_results:
        print(f" {scan_results['error']}")
        print("\nExpected directory structure:")
        print("data/")
        print("├── comparative_politics/")
        print("│   ├── syllabus.pdf")
        print("│   ├── lecture01.docx")
        print("│   └── readings/")
        print("├── international_relations/")
        print("└── political_theory/")
        return False

    print(f" Found {scan_results['total_classes']} classes with {scan_results['total_documents']} documents")
    print("\nClass breakdown:")

    for class_name, class_info in scan_results["classes"].items():
        print(f"\n📁 {class_name}")
        print(f"   Path: {class_info['folder_path']}")
        print(f"   Documents: {class_info['document_count']}")

        for doc in class_info["documents"]:
            print(f"   📄 {doc['name']} ({doc['size_mb']} MB) - {doc['extension']}")

    return True

def test_single_document_processing(doc_path: str):
    """Test processing a single document from your data"""
    file_path = Path(doc_path)

    if not file_path.exists():
        print(f"File not found: {doc_path}")
        return

    print(f"\nTesting Document Processing: {file_path.name}")
    print("=" * 60)

    processor = DocumentProcessor()

    # Extract text
    print("1. Extracting text...")
    text = processor.extract_text_from_file(file_path)
    print(f"   Extracted {len(text)} characters")

    if len(text) < 100:
        print("    Warning: Very little text extracted")
        print(f"   Sample: {text[:200]}")
        return

    # Show first part of extracted text
    print(f"   First 300 chars: {text[:300]}...")

    # Test hierarchical structure detection
    print("\n2. Detecting hierarchical structure...")
    structured = processor.extract_hierarchical_structure(text)
    print(f"   Found {len(structured)} structured elements")

    # Show sample structured elements
    hierarchy_levels = {}
    for item in structured[:10]:
        level = item['level']
        hierarchy_levels[level] = hierarchy_levels.get(level, 0) + 1
        if len(structured) <= 10 or item['level'] > 0:  # Show all if few items, or just structured content
            print(f"   Level {item['level']}: {item['content'][:60]}...")

    print(f"   Hierarchy distribution: {hierarchy_levels}")

    # Test chunking
    print("\n3. Creating chunks...")
    chunks = processor.intelligent_chunking(text)
    print(f"   Created {len(chunks)} chunks")

    chunk_sizes = [len(chunk) for chunk in chunks]
    print(f"   Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)//len(chunk_sizes)}")

    # Show first chunk
    if chunks:
        print(f"\n   First chunk ({len(chunks[0])} chars):")
        print("   " + chunks[0][:400].replace('\n', '\n   ') + "...")

    # Test full document processing
    print("\n4. Full document processing with metadata...")
    doc_chunks = processor.process_document(file_path, "test_class", "Test Professor")
    print(f"   Created {len(doc_chunks)} DocumentChunk objects")

    if doc_chunks:
        sample_chunk = doc_chunks[0]
        print(f"\n   Sample metadata:")
        metadata_sample = {k: v for k, v in sample_chunk.metadata.items()
                          if k not in ['text'] and v is not None}
        print("   " + json.dumps(metadata_sample, indent=6)[1:-1])

    return True

def test_embedding_generation(data_dir: str = "data"):
    """Test embedding generation on a small sample"""
    print("\nTesting Embedding Generation...")
    print("=" * 60)

    # Find first document in data directory
    data_path = Path(data_dir)
    test_doc = None

    for class_folder in data_path.iterdir():
        if class_folder.is_dir():
            for ext in ['.pdf', '.docx']:
                docs = list(class_folder.glob(f"*{ext}"))
                if docs:
                    test_doc = docs[0]
                    break
            if test_doc:
                break

    if not test_doc:
        print("No test document found in data directory")
        return

    print(f"Testing with: {test_doc}")

    processor = DocumentProcessor()

    # Process document
    chunks = processor.process_document(test_doc, "test_class", "Test Professor")

    if not chunks:
        print("No chunks created")
        return

    print(f"Processing {len(chunks)} chunks...")

    # Generate embeddings
    chunks_with_embeddings = processor.generate_embeddings(chunks[:3])  # Test with first 3 chunks

    if chunks_with_embeddings and chunks_with_embeddings[0].embedding:
        embedding_dim = len(chunks_with_embeddings[0].embedding)
        print(f"Successfully generated embeddings (dimension: {embedding_dim})")
        print(f"   Sample embedding (first 5 values): {chunks_with_embeddings[0].embedding[:5]}")
    else:
        print("Failed to generate embeddings")

def main():
    """Run comprehensive tests on your data directory"""
    print("AI Professor Platform - Data Processing Tests")
    print("=" * 60)

    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "documents"

    print(f"Testing data directory: {data_dir}")

    # Test 1: Directory scan
    if not test_data_directory_scan(data_dir):
        print("\n Directory scan failed. Please fix directory structure first.")
        return

    # Test 2: Single document processing
    print("\n" + "=" * 60)

    # Find first document to test with
    data_path = Path(data_dir)
    test_doc = None

    for class_folder in data_path.iterdir():
        if class_folder.is_dir():
            for ext in ['.pdf', '.docx']:
                docs = list(class_folder.glob(f"**/*{ext}"))
                if docs:
                    test_doc = docs[0]
                    break
            if test_doc:
                break

    if test_doc:
        test_single_document_processing(str(test_doc))
    else:
        print("No documents found for individual testing")

    # Test 3: Embedding generation
    print("\n" + "=" * 60)
    test_embedding_generation(data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Testing Summary:")
    print("If all tests passed, you're ready to run full processing")
    print("Run: python data_processor.py")
    print("Then test Streamlit app: streamlit run app.py")
    print("\n If any tests failed:")
    print("   - Check file formats (PDF/DOCX supported)")
    print("   - Ensure files contain readable text")
    print("   - Verify directory structure matches expected format")

if __name__ == "__main__":
    main()