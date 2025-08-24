"""
Pedagogical Strategy Engine

Handles teaching strategies, formatting approaches, and response generation logic
for the AI Professor Platform.
"""

import logging
from typing import Dict, List
from conversation_manager import ConversationContext

logger = logging.getLogger(__name__)


class PedagogicalEngine:
    """Manages pedagogical strategies and formatting approaches based on question types and context"""
    
    def __init__(self):
        self.formatting_strategies = self._initialize_formatting_strategies()
        self.teaching_strategies = self._initialize_teaching_strategies()
    
    def _initialize_formatting_strategies(self) -> Dict[str, str]:
        """Initialize question-type specific formatting strategies"""
        return {
            "definition": """
FORMAT AS CONCEPTUAL BREAKDOWN:
- Use distinct category headers (e.g., "Politics as Power Allocation:", "Federalism as Shared Governance:")
- Provide clear explanations under each category with concrete examples
- Show concept progression: Foundation → Building Block → Application → Implication
- End with thought map showing how concepts connect
- NO bullet points - use paragraph structure with clear category divisions. Make sure that the category divisions are in bold""",
            
            "application": """
FORMAT AS PROCESS GUIDE:
- Present step-by-step phases clearly labeled
- Show decision frameworks: "Consider these factors:", "If X, then Y"
- Provide real-world scenario walk-throughs
- Use template structures students can follow
- Include practical considerations at each step""",
            
            "analysis": """
FORMAT AS MULTI-PERSPECTIVE EXAMINATION:
- Present comparative frameworks (Perspective A vs Perspective B)
- Show cause-and-effect chains with clear connections
- Use "Looking at this through different lenses:" approach
- Provide evidence for each viewpoint
- Build analytical scaffolding step by step""",
            
            "clarification": """
FORMAT AS LAYERED EXPLANATION:
- Break complex ideas into digestible components
- Use progression: Surface level → Deeper understanding → Broader implications
- Show connections to previously discussed concepts
- Include analogies and concrete examples
- Build from familiar concepts to unfamiliar ones""",
            
            "synthesis": """
FORMAT AS BUILDING BLOCK ASSEMBLY:
- Present individual components first
- Show how components combine and interact
- Highlight trade-offs and considerations
- Provide creative frameworks for integration
- Encourage innovative connections between ideas""",
            
            "general": """
FORMAT AS STRUCTURED EXPLORATION:
- Use clear thematic organization
- Provide substantial content with examples
- Show relationships between ideas
- Build cumulative understanding
- Connect to course framework and prior learning"""
        }
    
    def _initialize_teaching_strategies(self) -> Dict[str, Dict[str, str]]:
        """Initialize depth-based teaching strategies for each question type"""
        return {
            "definition": {
                "surface": "Provide a clear, comprehensive definition with examples from course materials. Include context about why this concept is important. End with one thoughtful question about how this concept might apply to a related scenario.",
                "intermediate": "Give a thorough explanation connecting this concept to previously discussed topics. Include multiple examples and explain the relationships between concepts. Conclude by inviting them to explore how this concept interacts with other theories.",
                "deep": "Provide an in-depth explanation covering multiple dimensions of the concept. Show how it relates to the broader theoretical framework and previous discussions. End by asking them to consider implications or applications they find most significant."
            },
            "application": {
                "surface": "Explain the relevant theoretical framework clearly, then walk through a concrete example step-by-step. Show how the principles are applied in practice. End by asking what other situations this framework might apply to.",
                "intermediate": "Present the theory and demonstrate its application through multiple examples. Connect to concepts we've previously discussed. Conclude by inviting them to identify which aspects of the application they'd like to explore further.",
                "deep": "Provide comprehensive coverage of the theoretical framework and its applications. Show multiple approaches and their trade-offs. End by asking them to evaluate which approach they think would be most effective in a specific context."
            },
            "analysis": {
                "surface": "Present multiple perspectives with supporting evidence from course materials. Explain the analytical framework and show how experts approach this type of analysis. End by asking which aspect they find most compelling or want to examine further.",
                "intermediate": "Provide a thorough analysis covering different viewpoints and their supporting evidence. Connect to our previous discussions and show how this analysis fits into the broader course themes. Conclude by asking what additional factors they think should be considered.",
                "deep": "Give a comprehensive analytical treatment examining assumptions, evidence, and implications. Show how this connects to the theoretical frameworks we've covered. End by inviting them to consider what the most important implications are for policy or practice."
            },
            "clarification": {
                "surface": "Provide a clear, detailed explanation of the concept with examples. Break down any complex parts into understandable components. End by asking if there are specific aspects they'd like to explore more deeply.",
                "intermediate": "Give a thorough explanation that builds on our previous discussions. Show how this concept fits into the larger framework of ideas we've covered. Conclude by asking what connections they see to other topics.",
                "deep": "Provide comprehensive clarification that addresses the complexity of the concept. Show multiple perspectives and nuances. End by asking them to reflect on which aspects they find most significant or challenging."
            }
        }
    
    def get_question_specific_formatting(self, question_type: str, prompt: str) -> str:
        """Get specific formatting instructions based on question type and content"""
        return self.formatting_strategies.get(question_type, self.formatting_strategies["general"])
    
    def get_pedagogical_strategy(self, question_type: str, context: ConversationContext) -> str:
        """Generate teaching strategy based on question type and conversation context"""
        depth = context.conversation_depth
        engagement = context.student_engagement_level
        consecutive_minimal = context.consecutive_minimal_responses
        
        # Handle disengagement with different strategies
        if engagement == "disengaged" or consecutive_minimal >= 2:
            return """STUDENT APPEARS DISENGAGED - CHANGE APPROACH:
            - Use a completely different angle or example than previous responses
            - Start with something concrete and relatable (current events, personal experience)
            - Simplify the explanation and use more engaging, conversational tone
            - Ask a different type of question - perhaps more personal or practical
            - Consider switching to a related but different concept from course materials
            - Use vivid, specific examples rather than abstract theory"""
        
        # Get strategy based on question type and depth
        strategies = self.teaching_strategies.get(question_type, {})
        return strategies.get(depth, "Provide substantive information from course materials with clear explanations and examples. End with one thoughtful question that encourages deeper exploration.")
    
    def generate_lesson_instruction(self, current_lesson: str, lesson_content_structure: Dict) -> str:
        """Generate lesson-focused instruction for responses"""
        if current_lesson != "all" and lesson_content_structure.get("current_lesson_chunks"):
            return f"""
LESSON FOCUS PRIORITY (CRITICAL):
- The student is currently studying Lesson {current_lesson}
- 95% of your response should focus on Lesson {current_lesson} content
- Only 5% should reference foundational connections from previous lessons when pedagogically valuable
- NEVER reference future lessons or content the student hasn't reached yet
- When making connections, explicitly note they're building on "previously covered concepts"
"""
        return ""
    
    def generate_repetition_guidance(self, main_concept: str, alternative_chunks: List[Dict]) -> str:
        """Generate guidance when content repetition is detected"""
        if not alternative_chunks:
            return ""
        
        alternative_topics = [chunk["hierarchy_path"] for chunk in alternative_chunks if chunk.get("hierarchy_path")]
        alternative_lessons = [f"Lesson {chunk['lesson']}" for chunk in alternative_chunks if chunk.get("lesson")]
        
        return f"""
CONTENT VARIATION REQUIRED - This concept has been covered recently:
- DO NOT repeat the same examples or explanations from previous responses
- EXPLORE DIFFERENT ANGLES: Consider these related topics from course materials: {', '.join(alternative_topics[:3])}
- ALTERNATIVE LESSONS: Draw from {', '.join(alternative_lessons[:2])} for fresh examples
- Use a completely different approach or perspective on this concept
- Connect to different real-world applications or historical contexts"""
    
    def generate_lesson_context(self, current_lesson: str, available_lessons: List[int]) -> str:
        """Generate lesson progression context information"""
        if current_lesson == "all" or not available_lessons:
            return ""
        
        try:
            current_num = int(current_lesson)
            completed_lessons = [str(i) for i in range(1, current_num)]
            return f"""
LESSON PROGRESSION CONTEXT:
- Student is currently on Lesson {current_lesson}
- Previously completed lessons: {', '.join(completed_lessons) if completed_lessons else 'None (this is their first lesson)'}
- Do NOT reference content from lessons beyond Lesson {current_lesson}
- Build appropriately on prior lessons when relevant
- Focus primarily on Lesson {current_lesson} concepts"""
        except (ValueError, TypeError):
            return ""
    
    def build_conversation_history(self, messages: List[Dict]) -> str:
        """Build conversation history context for responses"""
        if not messages:
            return ""
        
        recent_messages = messages[-4:]  # Last 2 exchanges
        conversation_history = "\n\nRECENT CONVERSATION CONTEXT:\n"
        for msg in recent_messages:
            role = "STUDENT" if msg["role"] == "user" else "PROFESSOR"
            conversation_history += f"{role}: {msg['content'][:200]}...\n"
        
        return conversation_history
    
    def build_topics_context(self, topics_discussed: List[str]) -> str:
        """Build context of topics already discussed"""
        if not topics_discussed:
            return ""
        
        return f"\n\nTOPICS ALREADY DISCUSSED THIS SESSION:\n{', '.join(topics_discussed[-5:])}"