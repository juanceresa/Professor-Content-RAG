"""
Course Management Module

Handles course configuration, loading, and management for the AI Professor Platform.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CourseConfig:
    """Data class representing a course configuration"""
    key: str
    name: str
    namespace: str
    description: str
    system_prompt: str
    vector_count: int = 0


class CourseManager:
    """Manages course configurations and interactions with Pinecone namespaces"""
    
    def __init__(self):
        self._course_configs = self._load_course_definitions()
        self._available_courses = {}
        self._content_search = None
        
    def _load_course_definitions(self) -> Dict[str, CourseConfig]:
        """Load predefined course configurations"""
        course_definitions = {
            "federal_state_local": CourseConfig(
                key="federal_state_local",
                name="Federal, State, and Local Government",
                namespace="federal_state_local",
                description="American government systems at federal, state, and local levels",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in American Government with expertise in federal, state, and local systems as presented in your course materials."""
            ),
            "american": CourseConfig(
                key="american",
                name="American Political System",
                namespace="american",
                description="American political institutions, processes, and governance",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in American Politics as presented in your course materials."""
            ),
            "foundational": CourseConfig(
                key="foundational",
                name="Foundational Political Theory",
                namespace="foundational",
                description="Core concepts and foundations of political science",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in Foundational Political Theory as presented in your course materials."""
            ),
            "functional": CourseConfig(
                key="functional",
                name="Functional Political Analysis",
                namespace="functional",
                description="Functional approaches to understanding political systems",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in Functional Political Analysis as presented in your course materials."""
            ),
            "international": CourseConfig(
                key="international",
                name="International Relations & Comparative Politics",
                namespace="international",
                description="International relations, comparative politics, and global affairs",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in International Relations and Comparative Politics as presented in your course materials."""
            ),
            "professional": CourseConfig(
                key="professional",
                name="Professional & Management Politics",
                namespace="professional",
                description="Professional development and management in political contexts",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in Professional and Management Politics as presented in your course materials."""
            ),
            "theory": CourseConfig(
                key="theory",
                name="Political Philosophy & Theory",
                namespace="theory",
                description="Classical and modern political philosophy and theory",
                system_prompt="""You are Professor Robert Ceresa. You must respond using ONLY the language, terminology, concepts, and explanations found in your course materials.
                
                CRITICAL REQUIREMENTS:
                - Use only Professor Ceresa's specific terminology and conceptual frameworks from the course materials
                - Draw all explanations directly from the provided course content
                - Maintain Professor Ceresa's exact academic voice and pedagogical style
                - Never introduce concepts or language not present in the course materials
                - Write as if you are Professor Ceresa continuing your written course content
                
                You specialize in Political Philosophy and Theory, including classical and contemporary thought, as presented in your course materials."""
            )
        }
        return course_definitions
    
    def load_available_courses_from_pinecone(self, pinecone_index) -> Dict[str, CourseConfig]:
        """Load courses that exist in Pinecone based on predefined course mapping"""
        try:
            # Get index stats to see available namespaces
            stats = pinecone_index.describe_index_stats()
            existing_namespaces = stats.get('namespaces', {})

            # Only return courses that exist in Pinecone
            available_courses = {}
            for course_key, course_config in self._course_configs.items():
                # Check if namespace exists and has content
                if course_config.namespace in existing_namespaces:
                    vector_count = existing_namespaces[course_config.namespace].get('vector_count', 0)

                    if vector_count > 0:
                        # Course has content - add to available courses
                        course_config.vector_count = vector_count
                        available_courses[course_key] = course_config
                        logger.info(f"Found course: {course_config.name} ({vector_count} vectors)")

            self._available_courses = available_courses
            return available_courses

        except Exception as e:
            logger.error(f"Error loading courses from Pinecone: {e}")
            return {}
    
    def get_course(self, course_key: str) -> Optional[CourseConfig]:
        """Get a specific course configuration"""
        return self._available_courses.get(course_key)
    
    def get_all_available_courses(self) -> Dict[str, CourseConfig]:
        """Get all available courses"""
        return self._available_courses
    
    def get_course_options(self) -> List[tuple]:
        """Get course options for UI selection (key, name) tuples"""
        return [(key, config.name) for key, config in self._available_courses.items()]
    
    def get_available_lessons_for_course(self, content_search_engine, course_key: str) -> List[int]:
        """Get available lessons for a course from the single namespace"""
        try:
            course_config = self.get_course(course_key)
            if not course_config:
                return []

            # Use content search engine to get lessons directly from namespace
            lessons = content_search_engine.get_available_lessons(course_config.namespace)

            # Convert to integers for UI compatibility (handle any floats like 3.5)
            int_lessons = []
            for lesson in lessons:
                if isinstance(lesson, float) and lesson.is_integer():
                    int_lessons.append(int(lesson))
                elif isinstance(lesson, int):
                    int_lessons.append(lesson)
                else:
                    # Handle decimal lessons like 3.5
                    int_lessons.append(lesson)

            logger.info(f"Found lessons {int_lessons} for course {course_config.name}")
            return sorted(int_lessons)

        except Exception as e:
            logger.error(f"Error getting available lessons: {e}")
            return []