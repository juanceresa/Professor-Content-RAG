"""
Response Generator - Simplified

Orchestrates content search and response generation with format profiles.
"""

import logging
import re
from typing import Dict, List, Optional
from course_manager import CourseConfig
from response.lesson_tracker import LessonTracker
from response.pedagogical_engine import PedagogicalEngine
from response.content_search import ContentSearchEngine

logger = logging.getLogger(__name__)

# Fidelity guardrail: forbidden phrases that indicate off-topic content
FORBIDDEN_SNIPPETS = [
    "the study of", "political science", "hypotheses", "experiments",
    "methodology", "data collection", "research methods", "discipline",
    "as a field of study", "think of explanations as", "branch of the social sciences",
    "this course is designed", "this class treats", "we will explore"
]


class ResponseGenerator:
    """Orchestrates content search and response generation"""

    def __init__(
        self,
        content_search: ContentSearchEngine,
        pedagogical_engine: PedagogicalEngine,
        lesson_tracker: LessonTracker,
        genai_model,
    ):
        self.content_search = content_search
        self.pedagogical_engine = pedagogical_engine
        self.lesson_tracker = lesson_tracker
        self.genai_model = genai_model

    def generate_response(
        self,
        prompt: str,
        course_config: CourseConfig,
        messages: List[Dict],
        selected_lesson: Optional[str] = None,
    ) -> str:
        """Generate response with format profiles and fidelity guardrails"""
        try:
            # Update lesson selection if provided
            if selected_lesson is not None:
                self.lesson_tracker.update_lesson_selection(selected_lesson)

            # Get current lesson
            current_lesson = self.lesson_tracker.current_lesson

            # Debug: Log search attempt
            print(f"🔍 DEBUG: Searching for '{prompt}' in course '{course_config.namespace}', lesson '{current_lesson}'")

            # Search for relevant content
            search_results = self.content_search.search_course_content(
                query=prompt,
                course_namespace=course_config.namespace,
                conversation_context=None,  # Simplified - no complex context tracking
                current_lesson=current_lesson,
            )

            # Fallback for empty context
            search_results = search_results or {}
            search_results.setdefault("chunks", [])
            search_results.setdefault("context", "")

            if not search_results["context"].strip():
                logger.error(f"Empty search results for query: '{prompt}' in course: '{course_config.namespace}'")
                return f"I couldn't find relevant course material to answer your question. Please try rephrasing or ask about a different topic from the course."

            # Select format profile based on question
            format_profile = self.pedagogical_engine.select_format_profile(prompt)

            # Build conversation history context
            conversation_history = self._build_conversation_history(messages)

            # Build the complete prompt
            full_prompt = self._build_complete_prompt(
                system_prompt=course_config.system_prompt,
                context=search_results["context"],
                conversation_history=conversation_history,
                prompt=prompt,
                format_profile=format_profile,
                current_lesson=current_lesson,
            )

            # Generate response
            try:
                response = self.genai_model.generate_content(full_prompt)
                model_text = getattr(response, "text", None)
                if not model_text:
                    model_text = str(response)
            except Exception as gen_e:
                logger.error("Model generation failed: %s", gen_e)
                return "I hit an issue calling the model. Please try again, or rephrase your question."

            # Apply fidelity guardrails and post-processing
            final_response = self._finalize_response(model_text)

            return final_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

    def _build_conversation_history(self, messages: List[Dict]) -> str:
        """Build conversation history context for responses"""
        if not messages:
            return ""

        recent_messages = messages[-4:]  # Last 2 exchanges
        conversation_history = "\n\nRECENT CONVERSATION CONTEXT:\n"
        for msg in recent_messages:
            role = "STUDENT" if msg["role"] == "user" else "PROFESSOR"
            conversation_history += f"{role}: {msg['content'][:200]}...\n"

        return conversation_history

    def _build_complete_prompt(
        self,
        system_prompt: str,
        context: str,
        conversation_history: str,
        prompt: str,
        format_profile: Dict[str, str],
        current_lesson: str,
    ) -> str:
        """Build the complete prompt with format profiles"""
        # Sanitize context metadata
        sanitized_context = self._sanitize_context_metadata(context)

        pedagogical_instruction = f"""
CORE IDENTITY AND TASK:
You are Professor Robert Ceresa responding to a student. Your primary task is to provide educational responses using ONLY the language, terminology, concepts, examples, and explanations found in the course materials provided.

CRITICAL CONSTRAINTS - 100% COURSE FIDELITY:
1. **Source Material Fidelity:** Use ONLY Professor Ceresa's specific terminology, explanations, definitions, and conceptual frameworks from the course materials
2. **Style Consistency:** Adopt his exact writing style, tone, and pedagogical approach as demonstrated in the materials
3. **Content Boundaries:** Never introduce concepts, terms, or explanations not present in the course materials
4. **Voice Authenticity:** Write as if you are Professor Ceresa continuing his course materials

RESPONSE QUALITY FRAMEWORK:

DIRECT QUOTATION PRIORITY:
- When course materials contain direct definitions (e.g., "Politics is world building"), USE THOSE EXACT WORDS
- Start with Professor Ceresa's precise definition, then elaborate using other course content
- Never weaken or dilute his clear, direct statements into vague generalizations

INTELLIGENT RESPONSE FORMATTING (Markdown):
- Output valid **Markdown** following the format profile provided
- **Profile:** {format_profile.get('name')}
- **Goal:** {format_profile.get('goal')}
- **Directives:**
{format_profile.get('directives')}
- Use exemplar as style guide only: {format_profile.get('exemplar')}

MULTI-SOURCE INTEGRATION:
When multiple course materials are provided:
- USE PROFESSOR CERESA'S EXACT PHRASES AND DEFINITIONS when available in the materials
- Synthesize information into a coherent, logical explanation rather than dumping raw materials
- Look for the complete story across all provided materials
- Organize Professor Ceresa's ideas into a flowing, unified response
- Connect related concepts that appear in different sections

QUESTION-FOCUSED FILTERING:
Before including ANY sentence, ask: "Does this sentence help the student understand the answer to their exact question?"

RELEVANCE STANDARDS:
- ✅ DIRECTLY ANSWERS: Content that specifically addresses what the student asked
- ❌ COURSE META-CONTENT: Information about class design, pedagogical goals, course structure
- ❌ ACADEMIC CONTEXT: Disciplinary classifications, research methodologies, general academic theory
- ❌ TANGENTIAL CONTENT: Related concepts that don't directly answer the specific question

STOPPING RULES:
- Complete the answer to what was specifically asked using course materials
- Stop when the definition/process/comparison is complete
- Do not continue into course context, academic background, or tangentially related topics
- When you've thoroughly explained what the student asked about, STOP

DIRECT RESPONSE PROTOCOL:
- Start directly with the requested content without unnecessary introductory phrases
- Avoid filler affirmations like "Certainly!", "Of course!", "Absolutely!"
- Get straight to the substantive content the student is seeking

FINAL INSTRUCTION: Respond exactly as Professor Ceresa would, using only his voice, language, and conceptual frameworks from the course materials. Synthesize information into a coherent, unified explanation while remaining 100% faithful to his specific academic approach and terminology."""

        current_lesson_context = (
            "ALL course materials"
            if current_lesson == "all"
            else f"Lesson {current_lesson} materials"
        )

        full_prompt = f"""{pedagogical_instruction}

LESSON CONTEXT: You are responding based on {current_lesson_context}

COURSE MATERIALS CONTENT (Your ONLY source of knowledge):
{sanitized_context}

{conversation_history}

STUDENT QUESTION: {prompt}

RESPOND EXACTLY AS PROFESSOR CERESA WOULD:
Use only the language, concepts, examples, and explanations from the course materials above. Synthesize the information provided into a coherent, logical explanation that follows the format profile guidelines."""

        return full_prompt

    def _sanitize_context_metadata(self, context: str) -> str:
        """Clean up metadata while preserving content structure"""
        # Replace technical headers with cleaner section breaks
        sanitized = re.sub(r'=== COURSE MATERIAL \d+ ===.*?\n', '\n--- SECTION ---\n', context)

        # Remove source file references but keep topic information
        sanitized = re.sub(r'Source:.*?\n', '', sanitized)

        # Convert topic paths to readable headers when present
        sanitized = re.sub(r'Topic: (.*?)\n', r'TOPIC: \1\n', sanitized)

        # Remove technical equals separators but preserve content breaks
        sanitized = re.sub(r'={10,}', '---', sanitized)

        # Ensure proper paragraph spacing
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)

        # Clean up leading/trailing whitespace
        sanitized = sanitized.strip()

        # Add proper spacing around section breaks
        sanitized = re.sub(r'--- SECTION ---', '\n\n--- SECTION ---\n', sanitized)

        return sanitized.strip()

    def _violates_fidelity(self, text: str) -> bool:
        """Check if text contains off-topic or meta-content phrases"""
        low = text.lower()
        return any(kw in low for kw in FORBIDDEN_SNIPPETS)

    def _enforce_fidelity(self, text: str) -> str:
        """Remove paragraphs that violate course fidelity"""
        if not self._violates_fidelity(text):
            return text

        # Split into blocks and filter out violating ones
        blocks = [block.strip() for block in text.split("\n\n")]
        clean_blocks = [block for block in blocks if block and not self._violates_fidelity(block)]

        result = "\n\n".join(clean_blocks).strip()
        if not result:
            return "I can only provide information directly from Professor Ceresa's course materials."
        return result

    def _polish_markdown(self, text: str) -> str:
        """Light post-processing: fix spacing, tables, and code fences"""
        if not text:
            return text

        s = text.strip()

        # Collapse excessive blank lines
        s = re.sub(r"\n{3,}", "\n\n", s)

        # Close stray code fences
        if s.count("```") % 2 == 1:
            s += "\n```"

        # Ensure tables have separator lines
        s = re.sub(
            r"(\n\|[^|\n]+\|[^\n]*\n)(?!\|[ :-]+\|)",
            r"\1|---|---|\n", s
        )

        return s

    def _finalize_response(self, text: str) -> str:
        """Apply all post-processing steps"""
        text = self._enforce_fidelity(text)
        text = self._polish_markdown(text)
        return text
