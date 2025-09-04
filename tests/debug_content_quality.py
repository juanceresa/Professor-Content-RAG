"""
Debug Content Quality

Diagnoses the quality of chunking, embeddings, and search results to identify
why the AI Professor isn't understanding course content properly.
"""

import os
import sys
import toml
from pathlib import Path
import logging
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append('..')

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dual_content_handler import DualContentHandler
from content_search import ContentSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentQualityDebugger:
    """Debug content quality and search effectiveness"""
    
    def __init__(self):
        self.pinecone_index = None
        self.embedding_model = None
        self.dual_handler = None
        self.content_search = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize services from secrets.toml"""
        secrets_path = Path("../.streamlit/secrets.toml")
        
        if secrets_path.exists():
            try:
                secrets = toml.load(secrets_path)
                pinecone_api_key = secrets.get("PINECONE_API_KEY")
                pinecone_index_name = secrets.get("PINECONE_INDEX_NAME", "ai-professor-platform")
                logger.info("Loaded configuration from .streamlit/secrets.toml")
            except Exception as e:
                logger.error(f"Error reading secrets.toml: {e}")
                return
        else:
            logger.error("secrets.toml not found")
            return
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        # Initialize handlers
        self.dual_handler = DualContentHandler(self.pinecone_index)
        self.content_search = ContentSearchEngine(self.embedding_model, self.pinecone_index)
        
        logger.info("Services initialized successfully")
    
    def check_namespace_contents(self, course_name: str = "foundational"):
        """Check what content exists in course namespaces"""
        logger.info(f"=== CHECKING NAMESPACE CONTENTS FOR {course_name.upper()} ===")
        
        # Check if course has dual structure
        has_dual = self.dual_handler.has_dual_structure(course_name)
        logger.info(f"Course {course_name} has dual structure: {has_dual}")
        
        if has_dual:
            self._check_dual_namespaces(course_name)
        else:
            self._check_legacy_namespace(course_name)
    
    def _check_dual_namespaces(self, course_name: str):
        """Check dual structure namespaces"""
        mastery_ns = f"{course_name}-mastery"
        lessons_ns = f"{course_name}-lessons"
        
        logger.info(f"\n--- Checking {mastery_ns} namespace ---")
        self._sample_namespace_content(mastery_ns, "mastery")
        
        logger.info(f"\n--- Checking {lessons_ns} namespace ---")
        self._sample_namespace_content(lessons_ns, "lesson")
    
    def _check_legacy_namespace(self, course_name: str):
        """Check legacy single namespace"""
        logger.info(f"\n--- Checking {course_name} namespace ---")
        self._sample_namespace_content(course_name, "legacy")
    
    def _sample_namespace_content(self, namespace: str, content_type: str):
        """Sample content from a namespace"""
        try:
            # Get namespace stats
            stats = self.pinecone_index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if namespace not in namespaces:
                logger.warning(f"Namespace {namespace} does not exist!")
                return
            
            vector_count = namespaces[namespace].get('vector_count', 0)
            logger.info(f"Namespace {namespace}: {vector_count} vectors")
            
            if vector_count == 0:
                logger.warning(f"No vectors found in {namespace}")
                return
            
            # Sample some vectors
            sample_results = self.pinecone_index.query(
                vector=[0.0] * 768,  # Zero vector for diverse sampling
                top_k=min(5, vector_count),  # Sample up to 5
                include_metadata=True,
                namespace=namespace
            )
            
            logger.info(f"Sample content from {namespace}:")
            for i, match in enumerate(sample_results.matches):
                metadata = match.metadata
                text_preview = metadata.get('text', '')[:200] + "..." if len(metadata.get('text', '')) > 200 else metadata.get('text', '')
                
                logger.info(f"\n  Sample {i+1}:")
                logger.info(f"    Score: {match.score:.4f}")
                logger.info(f"    Content Type: {metadata.get('content_type', 'unknown')}")
                logger.info(f"    Lesson Number: {metadata.get('lesson_number', 'none')}")
                logger.info(f"    Source: {metadata.get('source', 'unknown')}")
                logger.info(f"    Text Preview: {text_preview}")
                
        except Exception as e:
            logger.error(f"Error sampling {namespace}: {e}")
    
    def test_lesson_search(self, course_name: str = "foundational", lesson: str = "1"):
        """Test searching for specific lesson content"""
        logger.info(f"\n=== TESTING LESSON SEARCH: {course_name.upper()} LESSON {lesson} ===")
        
        test_queries = [
            f"lesson {lesson}",
            f"what is lesson {lesson} about",
            f"explain lesson {lesson}",
            "politics",
            "government",
            "what is the main topic"
        ]
        
        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' ---")
            
            try:
                # Test dual search
                results = self.content_search.search_course_content_dual(
                    query=query,
                    course_namespace=course_name,
                    selected_lesson=lesson,
                    top_k=3
                )
                
                logger.info(f"Search strategy: {results.get('search_strategy', 'unknown')}")
                logger.info(f"Total found: {results.get('total_found', 0)}")
                
                chunks = results.get('chunks', [])
                if chunks:
                    logger.info(f"Found {len(chunks)} chunks:")
                    for i, chunk in enumerate(chunks):
                        text_preview = chunk.get('text', '')[:150] + "..." if len(chunk.get('text', '')) > 150 else chunk.get('text', '')
                        logger.info(f"  Chunk {i+1}: Score={chunk.get('score', 0):.4f}, Type={chunk.get('content_type', 'unknown')}")
                        logger.info(f"    Text: {text_preview}")
                else:
                    logger.warning("No chunks found!")
                    
            except Exception as e:
                logger.error(f"Error in search: {e}")
    
    def test_all_lessons_search(self, course_name: str = "foundational"):
        """Test searching with 'all lessons' selected"""
        logger.info(f"\n=== TESTING ALL LESSONS SEARCH: {course_name.upper()} ===")
        
        test_queries = [
            "what is politics",
            "define government", 
            "explain democracy",
            "main concepts in this course"
        ]
        
        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' (all lessons) ---")
            
            try:
                results = self.content_search.search_course_content_dual(
                    query=query,
                    course_namespace=course_name,
                    selected_lesson="all",
                    top_k=3
                )
                
                logger.info(f"Search strategy: {results.get('search_strategy', 'unknown')}")
                logger.info(f"Total found: {results.get('total_found', 0)}")
                
                chunks = results.get('chunks', [])
                if chunks:
                    logger.info(f"Found {len(chunks)} chunks:")
                    for i, chunk in enumerate(chunks):
                        text_preview = chunk.get('text', '')[:150] + "..." if len(chunk.get('text', '')) > 150 else chunk.get('text', '')
                        logger.info(f"  Chunk {i+1}: Score={chunk.get('score', 0):.4f}, Type={chunk.get('content_type', 'unknown')}")
                        logger.info(f"    Text: {text_preview}")
                else:
                    logger.warning("No chunks found!")
                    
            except Exception as e:
                logger.error(f"Error in search: {e}")
    
    def analyze_chunking_patterns(self, course_name: str = "foundational"):
        """Analyze patterns in how content was chunked"""
        logger.info(f"\n=== ANALYZING CHUNKING PATTERNS: {course_name.upper()} ===")
        
        has_dual = self.dual_handler.has_dual_structure(course_name)
        if not has_dual:
            logger.info("Course uses legacy structure - analyzing single namespace")
            self._analyze_legacy_chunking(course_name)
        else:
            logger.info("Course uses dual structure - analyzing both namespaces")
            self._analyze_dual_chunking(course_name)
    
    def _analyze_dual_chunking(self, course_name: str):
        """Analyze chunking in dual structure"""
        mastery_ns = f"{course_name}-mastery"
        lessons_ns = f"{course_name}-lessons"
        
        logger.info(f"\n--- Mastery chunking ({mastery_ns}) ---")
        self._analyze_namespace_chunking(mastery_ns)
        
        logger.info(f"\n--- Lessons chunking ({lessons_ns}) ---")
        self._analyze_namespace_chunking(lessons_ns)
    
    def _analyze_legacy_chunking(self, course_name: str):
        """Analyze chunking in legacy structure"""
        self._analyze_namespace_chunking(course_name)
    
    def _analyze_namespace_chunking(self, namespace: str):
        """Analyze chunking patterns in a namespace"""
        try:
            # Sample more content for analysis
            sample_results = self.pinecone_index.query(
                vector=[0.0] * 768,
                top_k=20,  # Get more samples
                include_metadata=True,
                namespace=namespace
            )
            
            if not sample_results.matches:
                logger.warning(f"No content found in {namespace}")
                return
            
            # Analyze patterns
            chunk_lengths = []
            content_types = {}
            lesson_numbers = {}
            sources = {}
            
            for match in sample_results.matches:
                metadata = match.metadata
                text = metadata.get('text', '')
                chunk_lengths.append(len(text))
                
                # Count content types
                content_type = metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Count lesson numbers
                lesson_num = metadata.get('lesson_number', 'none')
                lesson_numbers[lesson_num] = lesson_numbers.get(lesson_num, 0) + 1
                
                # Count sources
                source = metadata.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            # Report analysis
            avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            logger.info(f"  Average chunk length: {avg_length:.0f} characters")
            logger.info(f"  Chunk length range: {min(chunk_lengths)} - {max(chunk_lengths)}")
            logger.info(f"  Content types: {content_types}")
            logger.info(f"  Lesson distribution: {lesson_numbers}")
            logger.info(f"  Sources: {sources}")
            
        except Exception as e:
            logger.error(f"Error analyzing {namespace}: {e}")
    
    def run_full_diagnosis(self, course_name: str = "foundational"):
        """Run complete diagnosis"""
        logger.info(f"STARTING FULL DIAGNOSIS FOR {course_name.upper()}")
        logger.info("="*60)
        
        # Check what exists
        self.check_namespace_contents(course_name)
        
        # Test searches
        self.test_lesson_search(course_name, "1")
        self.test_all_lessons_search(course_name)
        
        # Analyze chunking
        self.analyze_chunking_patterns(course_name)
        
        logger.info("\n" + "="*60)
        logger.info("DIAGNOSIS COMPLETE")


def main():
    debugger = ContentQualityDebugger()
    
    # Test with foundational course
    debugger.run_full_diagnosis("foundational")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE - Check the logs above for issues!")


if __name__ == "__main__":
    main()