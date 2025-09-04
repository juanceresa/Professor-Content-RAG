"""
Dual Content Handler

Manages both legacy (single namespace) and new dual-indexing (mastery + lessons) 
content structures during the transition period.
"""

import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContentNamespaces:
    """Configuration for content search namespaces"""
    mastery: str  # Namespace for mastery/comprehensive knowledge
    lessons: Optional[str]  # Namespace for lesson-specific content (None for "all lessons")
    lesson_filter: Optional[Dict]  # Filter for specific lesson content


class DualContentHandler:
    """Handles content search for both legacy and dual-indexing course structures"""
    
    # Courses that have been updated to new dual structure
    DUAL_STRUCTURE_COURSES = {"foundational", "functional", "govt", "professional"}
    
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index
    
    def has_dual_structure(self, course_namespace: str) -> bool:
        """Check if a course uses the new dual-indexing structure"""
        return course_namespace in self.DUAL_STRUCTURE_COURSES
    
    def get_available_lessons(self, course_namespace: str) -> List[Union[int, float]]:
        """Get available lessons for a course, handling both old and new structures"""
        try:
            if self.has_dual_structure(course_namespace):
                return self._get_lessons_dual_structure(course_namespace)
            else:
                return self._get_lessons_legacy_structure(course_namespace)
        except Exception as e:
            logger.error(f"Error getting lessons for {course_namespace}: {e}")
            return []
    
    def _get_lessons_dual_structure(self, course_namespace: str) -> List[Union[int, float]]:
        """Get lessons from dual-structure course (lessons namespace)"""
        lessons_namespace = f"{course_namespace}-lessons"
        
        try:
            # Query the lessons namespace
            sample_results = self.pinecone_index.query(
                vector=[0.0] * 768,
                top_k=100,
                include_metadata=True,
                namespace=lessons_namespace,
                filter={"content_type": "lesson", "course": course_namespace}
            )
            
            # Extract lesson numbers
            lessons = set()
            for match in sample_results.matches:
                lesson_num = match.metadata.get('lesson_number', '')
                if lesson_num:
                    try:
                        if '.' in str(lesson_num):
                            lessons.add(float(lesson_num))
                        else:
                            lessons.add(int(lesson_num))
                    except (ValueError, TypeError):
                        continue
            
            return sorted(list(lessons))
            
        except Exception as e:
            logger.error(f"Error querying lessons namespace {lessons_namespace}: {e}")
            return []
    
    def _get_lessons_legacy_structure(self, course_namespace: str) -> List[int]:
        """Get lessons from legacy single-namespace structure"""
        try:
            # Query the main namespace for lesson metadata
            sample_results = self.pinecone_index.query(
                vector=[0.0] * 768,
                top_k=100,
                include_metadata=True,
                namespace=course_namespace,
                filter={"course": course_namespace}
            )
            
            # Extract lesson numbers
            lessons = set()
            for match in sample_results.matches:
                lesson_num = match.metadata.get('lesson_number', '')
                if lesson_num and str(lesson_num).isdigit():
                    lessons.add(int(lesson_num))
            
            return sorted(list(lessons))
            
        except Exception as e:
            logger.error(f"Error querying legacy namespace {course_namespace}: {e}")
            return []
    
    def get_content_namespaces(self, course_namespace: str, selected_lesson: str = "all") -> ContentNamespaces:
        """Get appropriate namespaces and filters for content search"""
        
        if self.has_dual_structure(course_namespace):
            return self._get_dual_structure_namespaces(course_namespace, selected_lesson)
        else:
            return self._get_legacy_structure_namespaces(course_namespace, selected_lesson)
    
    def _get_dual_structure_namespaces(self, course_namespace: str, selected_lesson: str) -> ContentNamespaces:
        """Get namespaces for dual-structure courses"""
        mastery_namespace = f"{course_namespace}-mastery"
        lessons_namespace = f"{course_namespace}-lessons"
        
        if selected_lesson == "all":
            # All lessons - use mastery knowledge only
            return ContentNamespaces(
                mastery=mastery_namespace,
                lessons=None,
                lesson_filter=None
            )
        else:
            # Specific lesson - use mastery + lesson boundaries
            return ContentNamespaces(
                mastery=mastery_namespace,
                lessons=lessons_namespace,
                lesson_filter={
                    "content_type": "lesson", 
                    "course": course_namespace,
                    "lesson_number": str(selected_lesson)
                }
            )
    
    def _get_legacy_structure_namespaces(self, course_namespace: str, selected_lesson: str) -> ContentNamespaces:
        """Get namespaces for legacy single-namespace courses"""
        if selected_lesson == "all":
            # All lessons - use main namespace without lesson filter
            return ContentNamespaces(
                mastery=course_namespace,
                lessons=None,
                lesson_filter=None
            )
        else:
            # Specific lesson - use main namespace with lesson filter
            return ContentNamespaces(
                mastery=course_namespace,
                lessons=course_namespace,
                lesson_filter={
                    "course": course_namespace,
                    "lesson_number": str(selected_lesson)
                }
            )
    
    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists in Pinecone"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            existing_namespaces = set(stats.get('namespaces', {}).keys())
            return namespace in existing_namespaces
        except Exception as e:
            logger.error(f"Error checking namespace {namespace}: {e}")
            return False
    
    def get_namespace_stats(self, course_namespace: str) -> Dict:
        """Get statistics for course namespaces"""
        stats = {"total_vectors": 0, "namespaces": {}}
        
        try:
            index_stats = self.pinecone_index.describe_index_stats()
            namespaces = index_stats.get('namespaces', {})
            
            if self.has_dual_structure(course_namespace):
                # Check dual structure namespaces
                mastery_ns = f"{course_namespace}-mastery"
                lessons_ns = f"{course_namespace}-lessons"
                
                for ns in [mastery_ns, lessons_ns]:
                    if ns in namespaces:
                        vector_count = namespaces[ns].get('vector_count', 0)
                        stats["namespaces"][ns] = vector_count
                        stats["total_vectors"] += vector_count
            else:
                # Check legacy namespace
                if course_namespace in namespaces:
                    vector_count = namespaces[course_namespace].get('vector_count', 0)
                    stats["namespaces"][course_namespace] = vector_count
                    stats["total_vectors"] = vector_count
            
        except Exception as e:
            logger.error(f"Error getting namespace stats for {course_namespace}: {e}")
        
        return stats