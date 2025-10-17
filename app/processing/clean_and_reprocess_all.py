"""
Clean and Re-process All Course Content

Comprehensive script to clean all namespaces and re-process with
improved chunking and metadata. Uses single namespace per course.
"""

import toml
import sys
import logging
from pathlib import Path
from typing import List, Set

# Add parent directory for imports
sys.path.append('..')

from pinecone import Pinecone
from embed_docs import process_course_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveReprocessor:
    """Handles complete cleanup and reprocessing of all course content"""

    def __init__(self):
        self.pinecone_index = None
        self.pinecone_api_key = None
        self.pinecone_index_name = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize services from secrets.toml"""
        secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"

        if secrets_path.exists():
            try:
                secrets = toml.load(secrets_path)
                self.pinecone_api_key = secrets.get("PINECONE_API_KEY")
                self.pinecone_index_name = secrets.get("PINECONE_INDEX_NAME", "ai-professor-platform")
                logger.info("Loaded configuration from .streamlit/secrets.toml")
                logger.info(f"API key length: {len(self.pinecone_api_key) if self.pinecone_api_key else 'None'}")
                logger.info(f"Index name: {self.pinecone_index_name}")
            except Exception as e:
                logger.error(f"Error reading secrets.toml: {e}")
                return
        else:
            logger.error(f"secrets.toml not found at: {secrets_path.absolute()}")
            return

        if not self.pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in secrets.toml")
            return

        # Initialize Pinecone
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            self.pinecone_index = pc.Index(self.pinecone_index_name)
            logger.info("Pinecone connection established successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            return

        logger.info("Services initialized successfully")
    
    def get_all_course_namespaces(self) -> Set[str]:
        """Get all course-related namespaces currently in Pinecone"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            existing_namespaces = set(stats.get('namespaces', {}).keys())

            # Filter for course namespaces (exclude system namespaces)
            course_namespaces = set()
            course_patterns = [
                'foundational', 'functional', 'professional',  # Single namespace courses
                'american', 'international', 'theory',  # Single namespace courses
                'federal_state_local', 'govt', 'local'  # Government courses (old and new)
            ]

            for namespace in existing_namespaces:
                # Check for exact matches and dual namespace patterns (old structure)
                for pattern in course_patterns:
                    if (namespace == pattern or
                        namespace.startswith(f"{pattern}-lessons") or
                        namespace.startswith(f"{pattern}-mastery")):
                        course_namespaces.add(namespace)
                        break

            return course_namespaces

        except Exception as e:
            logger.error(f"Error getting namespaces: {e}")
            return set()
    
    def clean_namespace(self, namespace: str) -> bool:
        """Clean a specific namespace"""
        try:
            logger.info(f"Cleaning namespace: {namespace}")
            
            # Check if namespace exists and has content
            stats = self.pinecone_index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if namespace not in namespaces:
                logger.info(f"Namespace {namespace} does not exist - skipping")
                return True
            
            vector_count = namespaces[namespace].get('vector_count', 0)
            if vector_count == 0:
                logger.info(f"Namespace {namespace} is already empty - skipping")
                return True
            
            logger.info(f"Deleting {vector_count} vectors from {namespace}")
            
            # Delete all vectors in the namespace
            self.pinecone_index.delete(delete_all=True, namespace=namespace)
            
            logger.info(f"✅ Successfully cleaned {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cleaning {namespace}: {e}")
            return False
    
    def clean_all_namespaces(self, namespaces: Set[str]) -> List[str]:
        """Clean all specified namespaces"""
        logger.info("=" * 60)
        logger.info("CLEANING ALL COURSE NAMESPACES")
        logger.info("=" * 60)
        
        cleaned_namespaces = []
        failed_namespaces = []
        
        for namespace in sorted(namespaces):
            if self.clean_namespace(namespace):
                cleaned_namespaces.append(namespace)
            else:
                failed_namespaces.append(namespace)
        
        logger.info(f"\\nCleaning Summary:")
        logger.info(f"✅ Successfully cleaned: {len(cleaned_namespaces)} namespaces")
        if cleaned_namespaces:
            for ns in cleaned_namespaces:
                logger.info(f"   - {ns}")
        
        if failed_namespaces:
            logger.error(f"❌ Failed to clean: {len(failed_namespaces)} namespaces")
            for ns in failed_namespaces:
                logger.error(f"   - {ns}")
        
        return cleaned_namespaces
    
    def reprocess_all_courses(self) -> dict:
        """Re-process all courses with improved chunking"""
        logger.info("=" * 60)
        logger.info("RE-PROCESSING ALL COURSES WITH IMPROVED CHUNKING")
        logger.info("=" * 60)

        data_directory = Path(__file__).parent.parent.parent / "documents"

        if not data_directory.exists():
            logger.error(f"Documents directory not found: {data_directory}")
            return {}

        # Process all courses using the simplified embed_docs function
        try:
            process_course_content(
                data_directory=data_directory,
                pinecone_api_key=self.pinecone_api_key,
                pinecone_index_name=self.pinecone_index_name,
                professor_name="Professor Robert Ceresa"
            )

            # Get updated stats to see what was created
            stats = self.pinecone_index.describe_index_stats()
            namespaces = stats.get('namespaces', {})

            logger.info("=" * 60)
            logger.info("RE-PROCESSING SUMMARY")
            logger.info("=" * 60)

            results = {}
            for namespace, info in namespaces.items():
                vector_count = info.get('vector_count', 0)
                if vector_count > 0:
                    logger.info(f"✅ {namespace}: {vector_count} vectors")
                    results[namespace] = True

            logger.info(f"\nTotal: {len(results)} courses processed")
            return results

        except Exception as e:
            logger.error(f"❌ Error processing courses: {e}")
            return {}
    
    def run_complete_cleanup_and_reprocessing(self):
        """Run the complete cleanup and reprocessing workflow"""
        logger.info("🧹 STARTING COMPLETE CLEANUP AND REPROCESSING")
        logger.info("=" * 60)
        
        # Step 1: Get all existing namespaces
        logger.info("Step 1: Discovering existing course namespaces...")
        namespaces = self.get_all_course_namespaces()
        
        if not namespaces:
            logger.warning("No course namespaces found to clean")
        else:
            logger.info(f"Found {len(namespaces)} course namespaces to clean:")
            for ns in sorted(namespaces):
                logger.info(f"  - {ns}")
        
        # Step 2: Clean all namespaces
        if namespaces:
            cleaned = self.clean_all_namespaces(namespaces)
            logger.info(f"\\n✅ Cleaned {len(cleaned)} namespaces")
        
        # Step 3: Re-process all courses
        logger.info("\\nStep 3: Re-processing all courses with improved chunking...")
        results = self.reprocess_all_courses()
        
        # Final summary
        logger.info("\\n" + "=" * 60)
        logger.info("🎉 COMPLETE CLEANUP AND REPROCESSING FINISHED")
        logger.info("=" * 60)
        
        successful_courses = [course for course, success in results.items() if success]
        if successful_courses:
            logger.info("✅ Successfully processed courses:")
            for course in successful_courses:
                logger.info(f"   - {course} (now has proper chunking and metadata)")
        
        failed_courses = [course for course, success in results.items() if not success]
        if failed_courses:
            logger.error("❌ Failed to process courses:")
            for course in failed_courses:
                logger.error(f"   - {course}")
        
        logger.info("\\n🔍 Next steps:")
        logger.info("1. Test lesson queries: 'explain lesson 1', 'what is lesson 2 about'")
        logger.info("2. Verify lesson boundary enforcement works properly")
        logger.info("3. Check that responses are coherent and stay within lesson scope")


def main():
    """Main function with safety confirmation"""
    
    print("⚠️  COMPREHENSIVE COURSE REPROCESSING")
    print("=" * 50)
    print("This script will:")
    print("1. Delete ALL existing course namespaces in Pinecone")
    print("2. Re-process ALL courses with improved chunking")
    print("3. Create clean namespaces with proper metadata")
    print("\\nThis process will take several minutes and cannot be undone.")
    
    confirm = input("\\nContinue with complete reprocessing? (type 'YES' to confirm): ")
    
    if confirm != 'YES':
        print("❌ Operation cancelled - no changes made")
        return
    
    print("\\n🚀 Starting comprehensive reprocessing...")
    
    # Run the complete workflow
    reprocessor = ComprehensiveReprocessor()
    reprocessor.run_complete_cleanup_and_reprocessing()
    
    print("\\n✅ Complete reprocessing finished!")
    print("Your AI Professor should now work properly with:")
    print("- Proper lesson boundary enforcement")
    print("- Meaningful, contextual chunks")
    print("- Accurate metadata")
    print("- No cross-lesson contamination")


if __name__ == "__main__":
    main()