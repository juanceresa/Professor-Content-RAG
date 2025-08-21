#!/usr/bin/env python3
"""
Auto-run document processing without user input
"""

from embed_docs import process_course_content
from pathlib import Path
import os

def main():
    print("Auto-running document processing...")
    
    # Configuration
    DATA_DIRECTORY = "documents"
    PROFESSOR_NAME = "Professor Robert Ceresa"
    
    # Get API keys from environment or use placeholder
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
    PINECONE_INDEX_NAME = "ai-professor-platform"
    
    print(f"Processing documents from: {DATA_DIRECTORY}")
    print(f"Professor: {PROFESSOR_NAME}")
    print(f"Pinecone index: {PINECONE_INDEX_NAME}")
    print("=" * 60)
    
    try:
        process_course_content(
            data_directory=Path(DATA_DIRECTORY),
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX_NAME,
            professor_name=PROFESSOR_NAME
        )
        print("\n🎉 Processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        print("Please check your API keys and try again.")

if __name__ == "__main__":
    main()