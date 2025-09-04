"""
Debug Foundational Course Processing

Test processing ONLY the foundational course to see exactly what happens
with lesson1.pdf vs lesson3.5.pdf
"""

import sys
import logging
from pathlib import Path

sys.path.append('../processing')
from dual_embed_processor import DualEmbedProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_foundational_processing():
    """Debug processing of foundational course specifically"""
    
    print("=" * 60)
    print("DEBUG FOUNDATIONAL COURSE PROCESSING")
    print("=" * 60)
    
    # Initialize processor
    import toml
    secrets_path = Path("../.streamlit/secrets.toml")
    secrets = toml.load(secrets_path)
    
    processor = DualEmbedProcessor(
        pinecone_api_key=secrets["PINECONE_API_KEY"],
        pinecone_index_name=secrets["PINECONE_INDEX_NAME"],
        professor_name="Professor Robert Ceresa"
    )
    
    # Test on foundational course directory
    foundational_dir = Path("../documents/foundational")
    
    print(f"\\nFoundational directory: {foundational_dir}")
    print(f"Directory exists: {foundational_dir.exists()}")
    
    if foundational_dir.exists():
        # List all lesson files
        lesson_files = list(foundational_dir.glob("lesson*.pdf"))
        print(f"\\nFound lesson files: {len(lesson_files)}")
        for f in sorted(lesson_files):
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        
        # Test lesson number extraction
        print(f"\\nTesting lesson number extraction:")
        for f in sorted(lesson_files):
            lesson_num = processor._extract_lesson_number(f.name)
            print(f"  - {f.name} -> lesson_number: '{lesson_num}'")
        
        # Process the course
        print(f"\\n" + "=" * 40)
        print("PROCESSING FOUNDATIONAL COURSE")
        print("=" * 40)
        
        success = processor.process_course(foundational_dir, "foundational")
        print(f"\\nProcessing result: {'✅ Success' if success else '❌ Failed'}")
    
    else:
        print(f"❌ Foundational directory not found: {foundational_dir}")


if __name__ == "__main__":
    debug_foundational_processing()