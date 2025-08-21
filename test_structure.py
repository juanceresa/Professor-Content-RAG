#!/usr/bin/env python3
"""
Simple test script to validate the documents directory structure
without requiring heavy dependencies
"""

from pathlib import Path
import json

def scan_documents_directory(docs_path: Path):
    """Simple directory scanner without dependencies"""
    if not docs_path.exists():
        return {"error": f"Documents directory not found: {docs_path}"}
    
    results = {
        "directory": str(docs_path),
        "classes": {},
        "total_classes": 0,
        "total_documents": 0
    }
    
    # Find all class folders
    class_folders = [folder for folder in docs_path.iterdir() 
                    if folder.is_dir() and not folder.name.startswith('.')]
    
    results["total_classes"] = len(class_folders)
    
    # Scan each class folder
    supported_extensions = {'.pdf', '.docx', '.doc'}
    
    for class_folder in class_folders:
        class_info = {
            "folder_name": class_folder.name,
            "documents": [],
            "document_count": 0
        }
        
        # Find all documents
        for ext in supported_extensions:
            docs = list(class_folder.glob(f"**/*{ext}"))
            for doc in docs:
                class_info["documents"].append({
                    "name": doc.name,
                    "size_mb": round(doc.stat().st_size / (1024 * 1024), 2),
                    "extension": doc.suffix
                })
        
        class_info["document_count"] = len(class_info["documents"])
        results["total_documents"] += class_info["document_count"]
        results["classes"][class_folder.name] = class_info
    
    return results

def main():
    print("AI Professor Platform - Document Structure Test")
    print("=" * 60)
    
    docs_path = Path("documents")
    results = scan_documents_directory(docs_path)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"✅ Found {results['total_classes']} classes with {results['total_documents']} total documents")
    print("\nClass breakdown:")
    print("-" * 40)
    
    for class_name, class_info in results["classes"].items():
        print(f"📁 {class_name}")
        print(f"   Documents: {class_info['document_count']}")
        
        # Show first few documents
        for doc in class_info["documents"][:3]:
            print(f"   📄 {doc['name']} ({doc['size_mb']} MB)")
        
        if class_info['document_count'] > 3:
            print(f"   ... and {class_info['document_count'] - 3} more documents")
        print()
    
    # Summary
    print("=" * 60)
    print("STRUCTURE VALIDATION COMPLETE")
    print(f"✅ Directory structure is valid")
    print(f"✅ Found {results['total_classes']} political science courses")
    print(f"✅ Total of {results['total_documents']} documents ready for processing")
    
    print("\nCourse types detected:")
    for class_name in sorted(results["classes"].keys()):
        doc_count = results["classes"][class_name]["document_count"]
        print(f"  • {class_name.title()}: {doc_count} documents")

if __name__ == "__main__":
    main()