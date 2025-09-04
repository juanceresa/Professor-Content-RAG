"""
Dual Embedding Processor

Handles processing course content into dual namespaces:
1. Mastery content (from learning/ directory) - full course knowledge
2. Lesson content (from lessonX.pdf files) - lesson boundaries

For courses with dual structure: foundational, functional, govt, professional
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
from dataclasses import dataclass

# Import existing components (adjust path since we're in processing/ subdirectory)
import sys
sys.path.append('..')
from embed_docs import DocumentProcessor, PineconeUploader, DocumentChunk

logger = logging.getLogger(__name__)


class DualEmbedProcessor:
    """Processes course content for dual-indexing structure"""

    # Courses using the new dual structure
    DUAL_STRUCTURE_COURSES = {"foundational", "functional", "govt", "professional"}

    def __init__(self, pinecone_api_key: str, pinecone_index_name: str, professor_name: str = "Professor"):
        self.doc_processor = DocumentProcessor()
        self.uploader = PineconeUploader(pinecone_api_key, pinecone_index_name)
        self.professor_name = professor_name

    def process_course(self, course_directory: Path, course_name: str) -> bool:
        """Process a single course, handling both dual and legacy structures"""
        try:
            if course_name in self.DUAL_STRUCTURE_COURSES:
                return self._process_dual_structure_course(course_directory, course_name)
            else:
                return self._process_legacy_course(course_directory, course_name)
        except Exception as e:
            logger.error(f"Error processing course {course_name}: {e}")
            return False

    def _process_dual_structure_course(self, course_directory: Path, course_name: str) -> bool:
        """Process course with dual structure (mastery + lessons)"""
        logger.info(f"Processing dual-structure course: {course_name}")

        # Process mastery content (learning directory)
        learning_dir = course_directory / "learning"
        if learning_dir.exists():
            mastery_success = self._process_mastery_content(learning_dir, course_name)
        else:
            logger.warning(f"No learning directory found for {course_name}")
            mastery_success = False

        # Process individual lessons
        lesson_success = self._process_lesson_content(course_directory, course_name)

        return mastery_success or lesson_success  # Success if either works

    def _process_mastery_content(self, learning_dir: Path, course_name: str) -> bool:
        """Process mastery content from learning directory"""
        logger.info(f"Processing mastery content for {course_name}")

        try:
            # Find all documents in learning directory
            document_files = []
            for pattern in ['*.pdf', '*.docx', '*.doc']:
                document_files.extend(learning_dir.glob(pattern))

            if not document_files:
                logger.warning(f"No documents found in {learning_dir}")
                return False

            all_chunks = []
            for doc_path in document_files:
                logger.info(f"Processing mastery document: {doc_path.name}")

                # Process document
                chunks = self.doc_processor.process_document(
                    file_path=doc_path,
                    course_name=course_name,
                    professor_name=self.professor_name
                )

                # Add mastery-specific metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        "content_type": "mastery",
                        "source_type": "learning_material"
                    })

                all_chunks.extend(chunks)

            if all_chunks:
                # Upload to mastery namespace
                mastery_namespace = f"{course_name}-mastery"
                self.uploader.upload_chunks(all_chunks, mastery_namespace)
                logger.info(f"Uploaded {len(all_chunks)} mastery chunks to {mastery_namespace}")
                return True

        except Exception as e:
            logger.error(f"Error processing mastery content for {course_name}: {e}")

        return False

    def _process_lesson_content(self, course_directory: Path, course_name: str) -> bool:
        """Process individual lesson files"""
        logger.info(f"Processing lesson content for {course_name}")

        try:
            # Find lesson files (lessonX.pdf, etc.)
            lesson_files = []
            for pattern in ['lesson*.pdf', 'lesson*.docx', 'lesson*.doc']:
                lesson_files.extend(course_directory.glob(pattern))

            if not lesson_files:
                logger.warning(f"No lesson files found in {course_directory}")
                return False

            all_chunks = []
            for lesson_path in sorted(lesson_files):  # Sort to ensure consistent processing
                lesson_number = self._extract_lesson_number(lesson_path.name)
                if lesson_number is None:
                    logger.warning(f"Could not extract lesson number from {lesson_path.name}")
                    continue

                logger.info(f"Processing lesson {lesson_number}: {lesson_path.name}")

                # Process lesson document
                chunks = self.doc_processor.process_document(
                    file_path=lesson_path,
                    course_name=course_name,
                    professor_name=self.professor_name
                )

                # Add lesson-specific metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        "content_type": "lesson",
                        "lesson_number": str(lesson_number),
                        "source_type": "lesson_material",
                        "lesson_file": lesson_path.name
                    })

                all_chunks.extend(chunks)

            if all_chunks:
                # Upload to lessons namespace
                lessons_namespace = f"{course_name}-lessons"
                self.uploader.upload_chunks(all_chunks, lessons_namespace)
                logger.info(f"Uploaded {len(all_chunks)} lesson chunks to {lessons_namespace}")
                return True

        except Exception as e:
            logger.error(f"Error processing lesson content for {course_name}: {e}")

        return False

    def _process_legacy_course(self, course_directory: Path, course_name: str) -> bool:
        """Process course with legacy single-namespace structure"""
        logger.info(f"Processing legacy course: {course_name}")

        try:
            # Find all documents in course directory (excluding subdirectories for now)
            document_files = []
            for pattern in ['*.pdf', '*.docx', '*.doc']:
                document_files.extend(course_directory.glob(pattern))

            if not document_files:
                logger.warning(f"No documents found in {course_directory}")
                return False

            all_chunks = []
            for doc_path in document_files:
                logger.info(f"Processing legacy document: {doc_path.name}")

                chunks = self.doc_processor.process_document(
                    file_path=doc_path,
                    course_name=course_name,
                    professor_name=self.professor_name
                )

                all_chunks.extend(chunks)

            if all_chunks:
                # Upload to single namespace (legacy)
                self.uploader.upload_chunks(all_chunks, course_name)
                logger.info(f"Uploaded {len(all_chunks)} chunks to legacy namespace {course_name}")
                return True

        except Exception as e:
            logger.error(f"Error processing legacy course {course_name}: {e}")

        return False

    def _extract_lesson_number(self, filename: str) -> Optional[str]:
        """Extract lesson number from filename like 'lesson3.pdf' or 'lesson3.5.pdf'"""
        import re

        # Handle patterns like lesson1.pdf, lesson3.5.pdf, etc.
        match = re.search(r'lesson(\d+(?:\.\d+)?)', filename.lower())
        if match:
            return match.group(1)

        return None

    def process_all_courses(self, data_directory: Path) -> Dict[str, bool]:
        """Process all courses in the data directory"""
        results = {}

        if not data_directory.exists():
            logger.error(f"Data directory not found: {data_directory}")
            return results

        logger.info(f"Processing all courses from: {data_directory}")

        # Process each subdirectory as a course
        for course_dir in data_directory.iterdir():
            if course_dir.is_dir() and not course_dir.name.startswith('.'):
                course_name = course_dir.name
                logger.info(f"Found course directory: {course_name}")

                success = self.process_course(course_dir, course_name)
                results[course_name] = success

                if success:
                    logger.info(f"✅ Successfully processed course: {course_name}")
                else:
                    logger.error(f"❌ Failed to process course: {course_name}")

        return results


def main():
    """Main function for processing course content with dual structure"""
    import sys
    import toml

    # Try to load from secrets.toml first, fallback to environment variables
    secrets_path = Path("../.streamlit/secrets.toml")  # Go up one level from processing/

    if secrets_path.exists():
        try:
            secrets = toml.load(secrets_path)
            pinecone_api_key = secrets.get("PINECONE_API_KEY")
            pinecone_index_name = secrets.get("PINECONE_INDEX_NAME", "ai-professor-platform")
            logger.info("Loaded configuration from .streamlit/secrets.toml")
        except Exception as e:
            logger.warning(f"Error reading secrets.toml: {e}, falling back to environment variables")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "ai-professor-platform")
    else:
        logger.info("No secrets.toml found, using environment variables")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "ai-professor-platform")

    # Configuration
    data_directory = Path("../documents")  # Go up one level from processing/
    professor_name = "Professor Robert Ceresa"

    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY is required. Set it in .streamlit/secrets.toml or as environment variable")
        sys.exit(1)

    # Initialize processor
    processor = DualEmbedProcessor(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        professor_name=professor_name
    )

    # Process all courses
    results = processor.process_all_courses(data_directory)

    # Report results
    logger.info("=== Processing Summary ===")
    for course_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"{course_name}: {status}")


if __name__ == "__main__":
    main()