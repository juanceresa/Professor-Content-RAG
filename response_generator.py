"""
Response Generator (Enhanced)

Orchestrates all components to generate pedagogical responses for the AI Professor Platform.
Implements format profiles, fidelity guardrails, and intelligent Markdown formatting.
"""

import logging
import re
from typing import Dict, List, Optional
from course_manager import CourseConfig
from conversation_manager import ConversationManager
from pedagogical_engine import PedagogicalEngine
from content_search import ContentSearchEngine

logger = logging.getLogger(__name__)

# Fidelity guardrail: forbidden phrases that indicate off-topic content
FORBIDDEN_SNIPPETS = [
    "the study of", "political science", "hypotheses", "experiments",
    "methodology", "data collection", "research methods", "discipline",
    "as a field of study", "think of explanations as", "branch of the social sciences",
    "this course is designed", "this class treats", "we will explore"
]


class ResponseGenerator:
    """Orchestrates content search, pedagogical strategies, and response generation"""

    def __init__(
        self,
        content_search: ContentSearchEngine,
        pedagogical_engine: PedagogicalEngine,
        conversation_manager: ConversationManager,
        genai_model,
    ):
        self.content_search = content_search
        self.pedagogical_engine = pedagogical_engine
        self.conversation_manager = conversation_manager
        self.genai_model = genai_model

    def generate_response(
        self,
        prompt: str,
        course_config: CourseConfig,
        messages: List[Dict],
        selected_lesson: Optional[str] = None,
    ) -> str:
        """Generate pedagogical response with format profiles and fidelity guardrails."""
        try:
            # 0) If caller provided a specific lesson, sync it (don't let failures crash response)
            if selected_lesson is not None:
                try:
                    self.conversation_manager.update_lesson_selection(selected_lesson)
                except Exception as e:
                    logger.warning(
                        "Failed to update lesson selection to %r: %s", selected_lesson, e
                    )

            # 1) Search for relevant content using dual indexing strategy
            current_lesson = self.conversation_manager.context.current_lesson
            
            # Try dual search first (for courses with dual structure)
            try:
                search_results = self.content_search.search_course_content_dual(
                    query=prompt,
                    course_namespace=course_config.namespace,
                    selected_lesson=current_lesson,
                    conversation_context=self.conversation_manager.context,
                )
                
                # Convert dual search results to expected format
                if search_results and search_results.get("chunks"):
                    # Format chunks into expected structure
                    formatted_chunks = []
                    for chunk in search_results["chunks"]:
                        formatted_chunks.append({
                            "text": chunk.get("text", ""),
                            "score": chunk.get("score", 0),
                            "lesson": chunk.get("lesson", ""),
                            "hierarchy_path": chunk.get("source", ""),
                            "document": chunk.get("source", ""),
                            "chunk_type": "general"
                        })
                    
                    # Format context from chunks
                    context_parts = []
                    for i, chunk in enumerate(formatted_chunks[:3]):  # Use top 3
                        context_part = f"=== COURSE MATERIAL {i+1} ==="
                        if chunk["lesson"]:
                            context_part += f" (Lesson {chunk['lesson']})"
                        context_part += f"\n\n{chunk['text']}"
                        context_parts.append(context_part)
                    
                    search_results = {
                        "chunks": formatted_chunks,
                        "context": "\n\n" + "="*60 + "\n\n".join(context_parts) + "\n\n" + "="*60,
                        "total_matches": search_results.get("total_found", len(formatted_chunks))
                    }
                else:
                    # Fallback to legacy search if dual search returns no results
                    search_results = self.content_search.search_course_content(
                        query=prompt,
                        course_namespace=course_config.namespace,
                        conversation_context=self.conversation_manager.context,
                        current_lesson=current_lesson,
                    )
                    
            except Exception as e:
                logger.warning(f"Dual search failed, falling back to legacy search: {e}")
                # Fallback to legacy search method
                search_results = self.content_search.search_course_content(
                    query=prompt,
                    course_namespace=course_config.namespace,
                    conversation_context=self.conversation_manager.context,
                    current_lesson=current_lesson,
                )

            # Fallback for empty context
            search_results = search_results or {}
            search_results.setdefault("chunks", [])
            search_results.setdefault("context", "")
            
            if not search_results["context"].strip():
                return "I don't have course materials available to answer this specific question. Please try rephrasing or ask about a different topic covered in the course."

            # 2) Analyze question and select format profile
            question_type = self.conversation_manager.analyze_question_type(prompt)
            format_profile = self._select_format_profile(question_type, prompt)
            
            # 3) Update conversation context
            self.conversation_manager.update_conversation_context(prompt, search_results)

            # 4) Get pedagogical strategies
            teaching_strategy = self.pedagogical_engine.get_pedagogical_strategy(
                question_type, self.conversation_manager.context
            )
            formatting_strategy = self.pedagogical_engine.get_question_specific_formatting(
                question_type, prompt
            )

            # 5) Analyze content structure
            current_chunks = search_results.get("chunks", [])
            lesson_content_structure = self.content_search.analyze_lesson_content_structure(
                current_chunks, self.conversation_manager.context.current_lesson
            )

            # 6) Build instruction components
            instruction_components = self._build_instruction_components(
                question_type=question_type,
                teaching_strategy=teaching_strategy,
                formatting_strategy=formatting_strategy,
                lesson_content_structure=lesson_content_structure,
                current_chunks=current_chunks,
                course_config=course_config,
                search_results=search_results,
                messages=messages,
                format_profile=format_profile,
            )

            # 7) Build the complete prompt
            full_prompt = self._build_complete_prompt(
                system_prompt=course_config.system_prompt,
                instruction_components=instruction_components,
                context=search_results["context"],
                messages=messages,
                prompt=prompt,
            )

            # 8) Generate response
            try:
                response = self.genai_model.generate_content(full_prompt)
                model_text = getattr(response, "text", None)
                if not model_text:
                    model_text = str(response)
            except Exception as gen_e:
                logger.error("Model generation failed: %s", gen_e)
                return "I hit an issue calling the model. Please try again, or rephrase your question."

            # 9) Apply fidelity guardrails and post-processing
            final_response = self._finalize_response(model_text)

            # 10) Track content usage & concepts
            self.conversation_manager.track_content_usage(current_chunks)
            if any(kw in prompt.lower() for kw in ("what is", "define", "explain")):
                words = prompt.split()
                concept = words[-1] if len(words) > 2 else "concept"
                self.conversation_manager.track_concept_introduction(concept)

            return final_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

    def _select_format_profile(self, question_type: str, prompt: str) -> Dict[str, str]:
        """Select appropriate format profile based on question type and content."""
        p = prompt.lower()
        
        # Comparison questions
        if question_type == "compare" or any(k in p for k in ["compare", " vs ", "difference between"]):
            return {
                "name": "comparison",
                "goal": "Clear, skimmable comparison",
                "directives": """\
- Start with a 1–2 sentence overview.
- Then a 2–6 row **Markdown table** for key aspects.
- Follow with **bulleted pros/cons** and a 1-line "Choose X if / Choose Y if".\
""",
                "exemplar": """\
### TL;DR
| Aspect | X | Y |
|---|---|---|
| Core idea | … | … |
| Best when | … | … |

- **X pros:** …
- **Y pros:** …

**Choose X if…** | **Choose Y if…**
"""
            }
        
        # Definition questions
        if question_type == "definition" or any(k in p for k in ["what is", "define"]):
            return {
                "name": "definition",
                "goal": "Crisp definition, then essentials",
                "directives": """\
- Open with a bold **Definition** line (1 sentence).
- 2–3 short paragraphs for nuance.
- If variants exist in-course, list as bullets.\
""",
                "exemplar": """\
**Definition:** …

**Key characteristics**
- …
- …
"""
            }
        
        # Process/how-to questions
        if question_type == "how" or any(k in p for k in ["how does", "process", "steps"]):
            return {
                "name": "process",
                "goal": "Ordered steps",
                "directives": """\
- One-paragraph overview.
- **Numbered list** of steps, each 1–2 lines.\
""",
                "exemplar": """\
**Overview:** …

1. Step — …
2. Step — …
3. Step — …
"""
            }
        
        # Default narrative
        return {
            "name": "narrative",
            "goal": "Readable academic prose",
            "directives": "- Use short paragraphs; add a sub-header only if the content naturally splits.",
            "exemplar": ""
        }

    def _violates_fidelity(self, text: str) -> bool:
        """Check if text contains off-topic or meta-content phrases."""
        low = text.lower()
        return any(kw in low for kw in FORBIDDEN_SNIPPETS)

    def _enforce_fidelity(self, text: str) -> str:
        """Remove paragraphs that violate course fidelity."""
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
        """Light post-processing: fix spacing, tables, and code fences."""
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
        """Apply all post-processing steps."""
        text = self._enforce_fidelity(text)
        text = self._polish_markdown(text)
        return text

    def _build_instruction_components(
        self,
        question_type: str,
        teaching_strategy: str,
        formatting_strategy: str,
        lesson_content_structure: Dict,
        current_chunks: List[Dict],
        course_config: CourseConfig,
        search_results: Dict,
        messages: List[Dict],
        format_profile: Dict[str, str],
    ) -> Dict:
        """Build all instruction components for the response"""
        current_lesson = self.conversation_manager.context.current_lesson

        lesson_instruction = self.pedagogical_engine.generate_lesson_instruction(
            current_lesson, lesson_content_structure
        )

        repetition_guidance = ""
        main_concept = ""
        if current_chunks:
            main_concept = current_chunks[0].get("hierarchy_path", "").lower()

        if main_concept and self.conversation_manager.detect_content_repetition(
            main_concept, main_concept
        ):
            alternative_chunks = self.content_search.get_alternative_content_from_embeddings(
                main_concept, current_chunks, course_config.namespace
            )
            repetition_guidance = self.pedagogical_engine.generate_repetition_guidance(
                main_concept, alternative_chunks
            )

        conversation_history = self.pedagogical_engine.build_conversation_history(messages)
        topics_context = self.pedagogical_engine.build_topics_context(
            self.conversation_manager.context.topics_discussed
        )

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
            "student_engagement": self.conversation_manager.context.student_engagement_level,
            "format_profile": format_profile,
        }

    def _build_complete_prompt(
        self,
        system_prompt: str,
        instruction_components: Dict,
        context: str,
        messages: List[Dict],
        prompt: str,
    ) -> str:
        """Build the complete prompt with format profiles and stopping rules."""
        sanitized_context = self.content_search.sanitize_context_metadata(context)
        format_profile = instruction_components["format_profile"]

        pedagogical_instruction = f"""
PROFESSOR CERESA'S TEACHING APPROACH:
You are Professor Robert Ceresa responding to a student. You must respond using ONLY the language, terminology, concepts, examples, and explanations found in the course materials provided below.

CRITICAL CONSTRAINTS - 100% COURSE FIDELITY:
1. Use ONLY Professor Ceresa's specific terminology from the course materials
2. Use ONLY his explanations, definitions, and conceptual frameworks
3. Use ONLY examples, analogies, and references found in the course content
4. Adopt his exact writing style, tone, and pedagogical approach as demonstrated in the materials
5. Never introduce concepts, terms, or explanations not present in the course materials
6. Never use generic textbook language or your pre-training knowledge
7. Write as if you are Professor Ceresa continuing his course materials

INTELLIGENT RESPONSE FORMATTING (Markdown):
- Output valid **Markdown**.
- Follow this format profile:
  - **Profile:** {format_profile.get('name')}
  - **Goal:** {format_profile.get('goal')}
  - **Directives:**
{format_profile.get('directives')}
- Use this as a *style cue* only (do not copy content verbatim):
{format_profile.get('exemplar')}

UNIVERSAL QUESTION-FOCUSED FILTER: 
Before including ANY sentence, ask "Does this sentence help the student understand the answer to their exact question?"

SENTENCE RELEVANCE TEST:
- ✅ DIRECTLY ANSWERS: Content that specifically addresses what the student asked
- ❌ COURSE META-CONTENT: Information about the class, course design, pedagogical goals
- ❌ ACADEMIC CONTEXT: Disciplinary classifications, research approaches, methodology
- ❌ TANGENTIAL THEORY: Related concepts that don't directly answer the specific question

RESPONSE BOUNDARY PRINCIPLE:
- Answer only the question asked using course materials.
- Do not include disciplinary methods, research procedures, or origins of beliefs.
- When the definition/process/comparison is complete, STOP.

CONCRETE STOPPING EXAMPLE (for "What is politics?"):
End at: "…engaging diverse views and interests for the purpose of accomplishing things of lasting public value. By 'world,' I mean society and/or the community."
Do NOT continue with "The study of politics…", "Think of explanations as hypotheses…".

UNIVERSAL STOPPING RULE: When you've thoroughly explained what the student specifically asked about, STOP. Do not continue into course context, academic background, or related topics.

SYNTHESIS REQUIREMENTS FOR MULTIPLE COURSE SECTIONS:
When multiple course materials are provided:
- DO NOT just dump or concatenate the raw materials
- SYNTHESIZE the information into a coherent, logical explanation
- Look for the COMPLETE story across all the provided materials
- Organize Professor Ceresa's ideas into a flowing, unified response
- Connect related concepts that appear in different sections

FORBIDDEN:
- Generic AI explanations not found in course materials
- Standard textbook definitions unless they appear in the course content
- Your own interpretations or paraphrases of concepts
- Examples or analogies not present in the professor's materials
- Generic academic language that doesn't match the professor's style
- Truncated responses or incomplete thoughts
- Random ellipses (...) or cutoffs
- Claims about what "we haven't covered" or "hasn't been addressed"
- Assumptions about lesson progression or what students have/haven't learned
- References to "these lessons" when in "All lessons" mode
- Following tangential threads that don't directly answer the question asked
- Including related but separate topics that would be better explored in different questions
- Going down rabbit holes of loosely connected ideas when a focused answer is needed
- Discussing methodology when asked about concepts
- Course design philosophy when explaining topics
- Origins/sources when asked for definitions (unless directly relevant to the definition)
- Academic classifications when explaining practical concepts
- Meta-discussions about studying/researching topics vs. explaining the topics themselves

REQUIRED: Respond as Professor Ceresa would, using only his voice, language, and conceptual frameworks from the course materials. Synthesize information into a coherent, unified explanation while remaining 100% faithful to his specific academic approach and terminology."""

        current_lesson_context = (
            "ALL course materials"
            if instruction_components["current_lesson"] == "all"
            else f"Lesson {instruction_components['current_lesson']} materials"
        )

        full_prompt = f"""{pedagogical_instruction}

LESSON CONTEXT: You are responding based on {current_lesson_context}

COURSE MATERIALS CONTENT (Your ONLY source of knowledge):
{sanitized_context}

CONVERSATION CONTEXT:
{instruction_components['conversation_history']}
{instruction_components['topics_context']}

STUDENT QUESTION: {prompt}

RESPOND EXACTLY AS PROFESSOR CERESA WOULD:
Use only the language, concepts, examples, and explanations from the course materials above. Synthesize the information provided into a coherent, logical explanation that follows the format profile guidelines. Do not make claims about what has or hasn't been covered - work with what you have been given. Write as if you are Professor Ceresa providing a comprehensive explanation of the topic."""

        return full_prompt