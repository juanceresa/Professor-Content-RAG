#!/usr/bin/env python3
"""
Debug document processing to see what went wrong
"""

from embed_docs import get_course_mapping, scan_data_directory
from pathlib import Path

def debug_processing():
    print("🔍 DEBUGGING DOCUMENT PROCESSING")
    print("=" * 60)
    
    # Check course mapping
    course_mapping = get_course_mapping()
    print(f"📋 Configured courses: {len(course_mapping)}")
    for key, config in course_mapping.items():
        print(f"   {key} → {config['name']} (folders: {config['folders']})")
    
    print("\n" + "="*60)
    
    # Check document directory
    docs_path = Path("documents")
    results = scan_data_directory(docs_path)
    
    if "error" in results:
        print(f"❌ Directory error: {results['error']}")
        return
    
    print(f"📁 Found document folders: {results['total_classes']}")
    print(f"📄 Total documents: {results['total_documents']}")
    
    print("\nFolder → Course mapping check:")
    print("-" * 40)
    
    for course_key, course_config in course_mapping.items():
        print(f"\n🤖 {course_config['name']}")
        print(f"   Course key: {course_key}")
        print(f"   Source folders: {course_config['folders']}")
        
        total_docs = 0
        found_folders = []
        missing_folders = []
        
        for folder_name in course_config['folders']:
            if folder_name in results['classes']:
                doc_count = results['classes'][folder_name]['document_count']
                total_docs += doc_count
                found_folders.append(f"{folder_name}({doc_count})")
            else:
                missing_folders.append(folder_name)
        
        if found_folders:
            print(f"   ✅ Found: {', '.join(found_folders)} = {total_docs} docs")
        if missing_folders:
            print(f"   ❌ Missing: {', '.join(missing_folders)}")
        if total_docs == 0:
            print(f"   ⚠️  NO DOCUMENTS FOUND")

if __name__ == "__main__":
    debug_processing()