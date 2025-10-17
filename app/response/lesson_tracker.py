"""
Lesson Tracker Module

Simple tracking of current lesson selection for the AI Professor Platform.
"""

from typing import List


class LessonTracker:
    """Tracks current lesson selection and available lessons"""

    def __init__(self):
        self.current_lesson = "all"
        self.available_lessons = []

    def update_lesson_selection(self, lesson: str):
        """Update current lesson selection"""
        self.current_lesson = lesson

    def set_available_lessons(self, lessons: List[int]):
        """Set available lessons for the current course"""
        self.available_lessons = lessons

    def reset(self):
        """Reset lesson tracking for new course/session"""
        self.current_lesson = "all"
        self.available_lessons = []
