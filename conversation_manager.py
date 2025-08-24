"""
Conversation Context Management Module

Handles tracking student engagement, learning progression, and conversation context
for the AI Professor Platform.
"""

import time
import logging
from typing import Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Data class representing conversation context state"""
    topics_discussed: List[str] = field(default_factory=list)
    concepts_introduced: List[str] = field(default_factory=list)
    student_understanding_signals: List[str] = field(default_factory=list)
    question_types: List[str] = field(default_factory=list)
    learning_progression: List[str] = field(default_factory=list)
    last_topic: str = None
    conversation_depth: str = "surface"  # surface, intermediate, deep
    teaching_mode: str = "guided_discovery"  # guided_discovery, socratic, direct_instruction
    hint_level: int = 0
    session_start_time: float = None
    examples_used: List[str] = field(default_factory=list)
    content_angles_explored: List[str] = field(default_factory=list)
    student_engagement_level: str = "neutral"  # engaged, neutral, disengaged
    consecutive_minimal_responses: int = 0
    concept_repetition_count: Dict[str, int] = field(default_factory=dict)
    preferred_learning_style: str = "unknown"  # concrete, theoretical, application-focused
    conversation_branch_history: List[str] = field(default_factory=list)
    current_lesson: str = "all"
    available_lessons: List[int] = field(default_factory=list)
    lesson_progression_history: List[Dict] = field(default_factory=list)


class ConversationManager:
    """Manages conversation context, student engagement tracking, and learning progression"""
    
    def __init__(self):
        self.context = ConversationContext()
        
    def reset_context(self):
        """Reset conversation context for new course/session"""
        self.context = ConversationContext()
        
    def analyze_student_engagement(self, response: str) -> str:
        """Analyze student response to determine engagement level"""
        response_lower = response.lower().strip()
        
        # Disengaged signals
        if response_lower in ["i don't know", "idk", "not sure", "no idea", "i'm confused", "what?"]:
            return "disengaged"
        
        # Minimal responses
        if len(response.split()) <= 3 and response_lower in ["yes", "no", "ok", "sure", "maybe", "i guess"]:
            return "minimal"
        
        # Engaged signals
        if any(phrase in response_lower for phrase in [
            "how does", "what about", "can you explain", "that's interesting", 
            "i think", "my understanding", "follow up", "another question"
        ]):
            return "engaged"
        
        # Moderate length responses
        if len(response.split()) > 10:
            return "engaged"
        elif len(response.split()) > 5:
            return "neutral"
        
        return "neutral"
    
    def analyze_question_type(self, question: str) -> str:
        """Analyze the type of question to determine teaching approach"""
        question_lower = question.lower()
        
        # Definition/concept questions
        if any(word in question_lower for word in ["what is", "define", "meaning of", "definition"]):
            return "definition"
        
        # Application questions (check specific "analyze" patterns first)
        elif any(phrase in question_lower for phrase in ["how do i analyze", "how to analyze"]):
            return "application"
        
        # Application questions (general)
        elif any(word in question_lower for word in ["how to", "apply", "use", "implement", "example"]):
            return "application"
        
        # Analysis questions (general "analyze" and other analysis terms)
        elif any(word in question_lower for word in ["analyze", "why", "compare", "contrast", "evaluate"]):
            return "analysis"
        
        # Synthesis questions
        elif any(word in question_lower for word in ["create", "design", "combine", "integrate"]):
            return "synthesis"
        
        # Clarification questions
        elif any(word in question_lower for word in ["explain", "clarify", "elaborate"]):
            return "clarification"
        
        return "general"
    
    def detect_content_repetition(self, topic: str, concept: str) -> bool:
        """Check if we've covered this concept too many times recently"""
        # Check if this specific concept has been repeated
        if concept in self.context.concept_repetition_count:
            if self.context.concept_repetition_count[concept] >= 2:
                return True
        
        # Check if we're stuck on the same topic
        recent_topics = self.context.topics_discussed[-3:]
        if recent_topics.count(topic) >= 2:
            return True
        
        return False
    
    def update_conversation_context(self, question: str, search_results: Dict):
        """Update conversation context based on current interaction"""
        # Initialize session start time if not set
        if self.context.session_start_time is None:
            self.context.session_start_time = time.time()
        
        # Analyze student engagement if this is a follow-up
        engagement = self.analyze_student_engagement(question)
        self.context.student_engagement_level = engagement
        
        # Track consecutive minimal responses
        if engagement in ["disengaged", "minimal"]:
            self.context.consecutive_minimal_responses += 1
        else:
            self.context.consecutive_minimal_responses = 0
        
        # Analyze and store question type
        question_type = self.analyze_question_type(question)
        self.context.question_types.append(question_type)
        
        # Extract topics from search results and track concept repetition
        if search_results.get("chunks"):
            for chunk in search_results["chunks"]:
                topic = chunk.get("hierarchy_path", "")
                lesson = chunk.get("lesson", "")
                
                if topic:
                    # Track concept repetition
                    concept_key = topic.lower()
                    if concept_key in self.context.concept_repetition_count:
                        self.context.concept_repetition_count[concept_key] += 1
                    else:
                        self.context.concept_repetition_count[concept_key] = 1
                    
                    # Add to topics discussed if not already there
                    if topic not in self.context.topics_discussed:
                        self.context.topics_discussed.append(topic)
                
                if lesson:
                    self.context.last_topic = f"Lesson {lesson}: {topic}"
        
        # Adjust conversation depth based on question types
        recent_questions = self.context.question_types[-3:]
        if len(recent_questions) >= 2:
            if all(q in ["analysis", "synthesis"] for q in recent_questions[-2:]):
                self.context.conversation_depth = "deep"
            elif any(q in ["application", "analysis"] for q in recent_questions):
                self.context.conversation_depth = "intermediate"
    
    def update_lesson_selection(self, lesson: str):
        """Update current lesson selection and track progression"""
        self.context.current_lesson = lesson
        # Add to progression history
        self.context.lesson_progression_history.append({
            "lesson": lesson,
            "timestamp": time.time()
        })
    
    def set_available_lessons(self, lessons: List[int]):
        """Set available lessons for the current course"""
        self.context.available_lessons = lessons
    
    def track_concept_introduction(self, concept: str):
        """Track that a concept has been introduced"""
        if concept not in self.context.concepts_introduced:
            self.context.concepts_introduced.append(concept)
    
    def track_content_usage(self, chunks: List[Dict]):
        """Track which content chunks have been used in responses"""
        for chunk in chunks:
            chunk_id = chunk.get('id', f"{chunk.get('hierarchy_path', '')}_{chunk.get('lesson', '')}")
            if chunk_id not in self.context.examples_used:
                self.context.examples_used.append(chunk_id)
    
    def get_context_summary(self) -> Dict:
        """Get a summary of current conversation context"""
        return {
            "topics_count": len(self.context.topics_discussed),
            "concepts_count": len(self.context.concepts_introduced),
            "question_types": list(set(self.context.question_types[-3:])),
            "engagement_level": self.context.student_engagement_level,
            "conversation_depth": self.context.conversation_depth,
            "current_lesson": self.context.current_lesson,
            "consecutive_minimal_responses": self.context.consecutive_minimal_responses
        }