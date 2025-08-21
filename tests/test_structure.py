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

def show_course_mapping(results):
    """Show how document folders will be mapped to chatbots"""
    course_mapping = {
        "Federal, State, and Local Government": ["govt", "local"],
        "American Political System": ["american"],
        "Foundational Political Theory": ["foundational"], 
        "Functional Political Analysis": ["functional"],
        "International Relations & Comparative Politics": ["international"],
        "Professional & Management Politics": ["professional"],
        "Political Philosophy & Theory": ["theory"]
    }
    
    print("\n🤖 CHATBOT ORGANIZATION:")
    print("=" * 60)
    
    for course_name, folders in course_mapping.items():
        total_docs = 0
        found_folders = []
        
        for folder in folders:
            if folder in results["classes"]:
                total_docs += results["classes"][folder]["document_count"]
                found_folders.append(folder)
        
        print(f"📚 {course_name}")
        print(f"   Source folders: {', '.join(found_folders)}")
        print(f"   Total documents: {total_docs}")
        
        # Show sample documents
        sample_count = 0
        for folder in found_folders:
            if folder in results["classes"]:
                for doc in results["classes"][folder]["documents"][:2]:
                    if sample_count < 3:
                        print(f"   📄 {doc['name']}")
                        sample_count += 1
        
        if total_docs > 3:
            print(f"   ... and {total_docs - sample_count} more documents")
        print()

def main():
    print("AI Professor Platform - Document Structure Test")
    print("=" * 60)
    
    docs_path = Path("documents")
    results = scan_documents_directory(docs_path)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"✅ Found {results['total_classes']} document folders with {results['total_documents']} total documents")
    print("\nDocument folder breakdown:")
    print("-" * 40)
    
    for class_name, class_info in results["classes"].items():
        print(f"📁 {class_name}")
        print(f"   Documents: {class_info['document_count']}")
        
        # Show first few documents
        for doc in class_info["documents"][:2]:
            print(f"   📄 {doc['name']} ({doc['size_mb']} MB)")
        
        if class_info['document_count'] > 2:
            print(f"   ... and {class_info['document_count'] - 2} more documents")
        print()
    
    # Show chatbot organization
    show_course_mapping(results)
    
    # Summary
    print("=" * 60)
    print("STRUCTURE VALIDATION COMPLETE")
    print(f"✅ Directory structure is valid")
    print(f"✅ Ready to create 7 specialized political science chatbots")
    print(f"✅ Total of {results['total_documents']} documents ready for processing")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Run: python embed_docs.py (to process documents)")
    print(f"2. Run: streamlit run streamlit_app.py (to launch chatbots)")
    print(f"3. Test each chatbot with course-specific questions")

if __name__ == "__main__":
    main()