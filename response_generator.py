"""
Response Generator

Orchestrates all components to generate pedagogical responses for the AI Professor Platform.
"""

import logging
from typing import Dict, List
from course_manager import CourseConfig
from conversation_manager import ConversationManager
from pedagogical_engine import PedagogicalEngine
from content_search import ContentSearchEngine

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Orchestrates content search, pedagogical strategies, and response generation"""
    
    def __init__(self, content_search: ContentSearchEngine, 
                 pedagogical_engine: PedagogicalEngine,
                 conversation_manager: ConversationManager,
                 genai_model):
        self.content_search = content_search
        self.pedagogical_engine = pedagogical_engine
        self.conversation_manager = conversation_manager
        self.genai_model = genai_model
    
    def generate_response(self, prompt: str, course_config: CourseConfig, 
                         messages: List[Dict]) -> str:
        """Generate pedagogical response using conversation context and teaching strategies"""
        try:
            # Search for relevant content
            search_results = self.content_search.search_course_content(
                query=prompt,
                course_namespace=course_config.namespace,
                conversation_context=self.conversation_manager.context,
                current_lesson=self.conversation_manager.context.current_lesson
            )
            
            # Update conversation context based on current interaction
            question_type = self.conversation_manager.analyze_question_type(prompt)
            self.conversation_manager.update_conversation_context(prompt, search_results)
            
            # Get pedagogical strategy
            teaching_strategy = self.pedagogical_engine.get_pedagogical_strategy(
                question_type, self.conversation_manager.context
            )
            
            # Get formatting strategy
            formatting_strategy = self.pedagogical_engine.get_question_specific_formatting(
                question_type, prompt
            )
            
            # Analyze content structure for lesson focus
            current_chunks = search_results.get("chunks", [])
            lesson_content_structure = self.content_search.analyze_lesson_content_structure(
                current_chunks, self.conversation_manager.context.current_lesson
            )
            
            # Generate all instruction components
            instruction_components = self._build_instruction_components(
                question_type=question_type,
                teaching_strategy=teaching_strategy,
                formatting_strategy=formatting_strategy,
                lesson_content_structure=lesson_content_structure,
                current_chunks=current_chunks,
                course_config=course_config,
                search_results=search_results,
                messages=messages
            )
            
            # Build the complete prompt
            full_prompt = self._build_complete_prompt(
                system_prompt=course_config.system_prompt,
                instruction_components=instruction_components,
                context=search_results["context"],
                messages=messages,
                prompt=prompt
            )
            
            # Generate response using Google AI
            response = self.genai_model.generate_content(full_prompt)
            
            # Track content usage for repetition detection
            self.conversation_manager.track_content_usage(current_chunks)
            
            # Track concepts introduced
            if any(word in prompt.lower() for word in ["what is", "define", "explain"]):
                concept = prompt.split()[-1] if len(prompt.split()) > 2 else "concept"
                self.conversation_manager.track_concept_introduction(concept)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def _build_instruction_components(self, question_type: str, teaching_strategy: str,
                                    formatting_strategy: str, lesson_content_structure: Dict,
                                    current_chunks: List[Dict], course_config: CourseConfig,
                                    search_results: Dict, messages: List[Dict]) -> Dict:
        """Build all instruction components for the response"""
        current_lesson = self.conversation_manager.context.current_lesson
        
        # Generate lesson instruction
        lesson_instruction = self.pedagogical_engine.generate_lesson_instruction(
            current_lesson, lesson_content_structure
        )
        
        # Check for content repetition and generate guidance
        repetition_guidance = ""
        main_concept = ""
        if current_chunks:
            main_concept = current_chunks[0].get("hierarchy_path", "").lower()
            
        if main_concept and self.conversation_manager.detect_content_repetition(main_concept, main_concept):
            alternative_chunks = self.content_search.get_alternative_content_from_embeddings(
                main_concept, current_chunks, course_config.namespace
            )
            repetition_guidance = self.pedagogical_engine.generate_repetition_guidance(
                main_concept, alternative_chunks
            )
        
        # Build conversation history context
        conversation_history = self.pedagogical_engine.build_conversation_history(messages)
        
        # Build topics context
        topics_context = self.pedagogical_engine.build_topics_context(
            self.conversation_manager.context.topics_discussed
        )
        
        # Build lesson context
        lesson_context = self.pedagogical_engine.generate_lesson_context(
            current_lesson, self.conversation_manager.context.available_lessons
        )
        
        return {
            "question_type": question_type,
            "teaching_strategy": teaching_strategy,
            "formatting_strategy": formatting_strategy,
            "lesson_instruction": lesson_instruction,
            "repetition_guidance": repetition_guidance,
            "conversation_history": conversation_history,
            "topics_context": topics_context,
            "lesson_context": lesson_context,
            "current_lesson": current_lesson,
            "conversation_depth": self.conversation_manager.context.conversation_depth,
            "student_engagement": self.conversation_manager.context.student_engagement_level
        }
    
    def _build_complete_prompt(self, system_prompt: str, instruction_components: Dict,
                             context: str, messages: List[Dict], prompt: str) -> str:
        """Build the complete prompt for the AI model"""
        # Sanitize context for student consumption
        sanitized_context = self.content_search.sanitize_context_metadata(context)
        
        pedagogical_instruction = f"""
BALANCED TEACHING APPROACH FOR THIS RESPONSE:
- Question Type: {instruction_components['question_type']}
- Conversation Depth: {instruction_components['conversation_depth']}
- Student Engagement: {instruction_components['student_engagement']}
- Current Lesson: {instruction_components['current_lesson']}
- Teaching Strategy: {instruction_components['teaching_strategy']}
- Formatting Approach: {instruction_components['formatting_strategy']}
{instruction_components['lesson_context']}
{instruction_components['repetition_guidance']}

RESPONSE STRUCTURE (Follow this order):
1. INFORMATIVE CONTENT (70-80% of response):
   - Provide substantial, clear information from course materials
   - Give comprehensive explanations with concrete examples
   - Include relevant context and why this topic matters
   - Connect to course frameworks and theories

2. STRATEGIC ENGAGEMENT (20-30% of response):
   - End with ONE thoughtful question that encourages deeper thinking
   - Invite exploration of applications, implications, or connections

FORMATTING STRATEGIES BY QUESTION TYPE:

1. DEFINITION/CONCEPTUAL QUESTIONS:
   - Use distinct category headers (e.g., "Politics as Power Allocation:", "Politics as World Building:")
   - Provide clear explanations under each category
   - Include concrete examples within each section
   - Show concept progression with arrows (→) and connecting phrases
   - End with thought progression map showing how concepts build on each other

2. APPLICATION QUESTIONS ("How do I..." / "How to..."):
   - Use step-by-step process format with clear phases
   - Show decision trees: "If this, then that" structures
   - Provide framework templates that students can follow
   - Include real-world scenario walk-throughs
   - Use "Consider these factors:" approach

3. ANALYSIS QUESTIONS ("Why..." / "Analyze..." / "Compare..."):
   - Use comparative frameworks (Side A vs Side B)
   - Present multiple perspectives with evidence for each
   - Show cause-and-effect chains
   - Use "Looking at this through different lenses:" approach
   - Provide analytical scaffolding

4. CLARIFICATION QUESTIONS ("Explain..." / "Elaborate..."):
   - Break complex ideas into digestible components
   - Use layered explanations (surface → deeper → implications)
   - Show connections to previously discussed concepts
   - Use analogies and concrete examples
   - Build from familiar to unfamiliar concepts

5. SYNTHESIS QUESTIONS ("Create..." / "Design..." / "Integrate..."):
   - Present building blocks and show how they combine
   - Use design thinking approaches
   - Show trade-offs and considerations
   - Provide creative frameworks
   - Encourage innovative connections between ideas

TEACHING GUIDELINES:
1. INFORM FIRST: Give students substantive content they can digest and learn from
2. Reference specific course materials, lessons, and examples
3. Build on concepts discussed in previous conversation when relevant
4. Use clear, academic language appropriate for university students
5. Provide enough detail for thorough understanding
6. End with strategic questioning to encourage further exploration
7. Avoid excessive questioning - one good follow-up question is sufficient

ADVANCED FORMATTING TECHNIQUES:
1. CONCEPT FLOW VISUALIZATION:
   - Use "🔄" for cyclic processes
   - Use "⚖️" for balanced/opposing concepts
   - Use "📈" for progressive/escalating ideas
   - Use "🎯" for focused outcomes or goals
   
2. THOUGHT PROGRESSION MAPS:
   - Foundation Concept → Building Block → Application → Implication
   - Problem Identification → Analysis → Solutions → Evaluation
   - Historical Context → Current State → Future Trends
   
3. RELATIONSHIP INDICATORS:
   - "This builds on..." (cumulative learning)
   - "In contrast to..." (comparative analysis)
   - "Because of this..." (causal relationships)
   - "This connects to..." (interdisciplinary links)
   
4. LEARNING SCAFFOLDS:
   - Start with familiar concepts
   - Introduce one new element at a time
   - Show how new concepts relate to known ones
   - Provide bridges between abstract and concrete

CONVERSATION CONTINUITY:
- Connect new information to previous discussions naturally
- Reference earlier concepts to build cumulative knowledge
- Adjust detail level based on demonstrated understanding

CRITICAL: Write ONLY your response as Professor Ceresa. Do NOT include any instructional text, placeholders like [mention a specific example], meta-commentary, or references to "this demonstration". Give a complete, polished academic response."""
        
        full_prompt = f"""{system_prompt}

{instruction_components['lesson_instruction']}

{pedagogical_instruction}

COURSE MATERIALS:
{sanitized_context}
{instruction_components['conversation_history']}
{instruction_components['topics_context']}

CURRENT STUDENT QUESTION: {prompt}

Respond as Professor Robert Ceresa with a complete, polished academic response. Provide substantive information first, then end with one engaging question. Do not include any instructional text, placeholders, or meta-commentary in your response."""

        return full_prompt