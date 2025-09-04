import logging
import re
from typing import Dict, Optional, List
from conversation_manager import ConversationContext

logger = logging.getLogger(__name__)


class ContentSearchEngine:
    """Manages content search, filtering, and ranking operations"""

    def __init__(self, embedding_model, pinecone_index):
        self.embedding_model = embedding_model
        self.pinecone_index = pinecone_index

    def search_course_content(
        self,
        query: str,
        course_namespace: str,
        conversation_context: ConversationContext,
        current_lesson: str = "all",
        top_k: int = 8,
        metadata_filter: Optional[dict] = None,
    ) -> Dict:
        """Search for relevant content with lesson-aware filtering and enhanced academic structure preservation."""
        try:
            # Enhanced query with conversation context
            context_enhanced_query = self._enhance_query_with_context(query, conversation_context)

            # Generate query embedding with context enhancement
            query_embedding = self.embedding_model.encode([context_enhanced_query])[0].tolist()

            # Build base filter always containing course
            base_filter = {"course": course_namespace}
            if metadata_filter:
                base_filter.update(metadata_filter)

            # Lesson-aware filter: if a specific lesson is set, add it
            if current_lesson != "all":
                base_filter["lesson"] = str(current_lesson)

            # For "all lessons", get more results to ensure comprehensive coverage
            search_top_k = top_k if current_lesson != "all" else top_k * 4

            # Search Pinecone with combined filter
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=search_top_k,
                include_metadata=True,
                namespace=course_namespace,
                filter=base_filter,
            )

            if not results.matches:
                return {"context": "No relevant course material found for this query.", "chunks": []}

            # Process and rank results
            processed_chunks = self._process_search_results(results, current_lesson, conversation_context)

            if not processed_chunks.get("selected_chunks"):
                return {"context": "No sufficiently relevant course material found for this query.", "chunks": []}

            # Format context for response generation
            formatted_context = self._format_context_for_response(processed_chunks["selected_chunks"])

            return {
                "context": formatted_context,
                "chunks": processed_chunks["selected_chunks"],
                "total_matches": len(results.matches),
            }

        except Exception as e:
            logger.error(f"Error searching course content: {e}")
            return {"context": f"Error retrieving course content: {str(e)}", "chunks": []}

    def _enhance_query_with_context(self, query: str, conversation_context: ConversationContext) -> str:
        """Enhance query with conversation context and definitional search terms for better search relevance"""
        context_enhanced_query = query

        # For definitional questions, add search terms to find definitions and core explanations
        definitional_triggers = ["what is", "define", "definition of", "meaning of", "explain"]
        if any(trigger in query.lower() for trigger in definitional_triggers):
            # Extract the main concept being asked about
            concept = query.lower()
            for trigger in definitional_triggers:
                concept = concept.replace(trigger, "").strip()
            # Add terms that help find definitional content
            context_enhanced_query = f"{query} {concept} definition meaning explanation concept"

        # Add previously discussed topics to enhance search relevance
        if conversation_context.topics_discussed:
            recent_topics = conversation_context.topics_discussed[-2:]
            if recent_topics:
                context_enhanced_query = f"{context_enhanced_query} {' '.join(recent_topics)}"

        return context_enhanced_query

    # NOTE: methods `_process_search_results` and `_format_context_for_response` should already exist
    # in your implementation. Ensure they handle the structures you pass here.

    def _process_search_results(self, results, current_lesson: str,
                              conversation_context: ConversationContext) -> Dict:
        """Process and rank search results with lesson-aware filtering"""
        # Group chunks by lesson accessibility
        lesson_appropriate_chunks = []
        future_lesson_chunks = []
        general_chunks = []

        # Determine accessible lessons
        accessible_lessons = self._get_accessible_lessons(current_lesson)

        # Categorize chunks
        for match in results.matches:
            chunk_info = self._extract_chunk_info(match)

            if not chunk_info["lesson"]:
                if match.score > 0.45:
                    general_chunks.append(chunk_info)
            elif accessible_lessons is None:
                if match.score > 0.45:
                    lesson_appropriate_chunks.append(chunk_info)
            else:
                try:
                    lesson_num = int(chunk_info["lesson"])
                    if lesson_num in accessible_lessons:
                        lesson_appropriate_chunks.append(chunk_info)
                    else:
                        future_lesson_chunks.append(chunk_info)
                except (ValueError, TypeError):
                    if match.score > 0.45:
                        general_chunks.append(chunk_info)

        # Sort chunks by lesson and academic value
        lesson_appropriate_chunks.sort(
            key=lambda chunk: self._calculate_chunk_score(chunk, current_lesson, conversation_context),
            reverse=True
        )
        general_chunks.sort(
            key=lambda chunk: self._calculate_chunk_score(chunk, current_lesson, conversation_context),
            reverse=True
        )

        # Select final chunks
        selected_chunks = self._select_final_chunks(
            lesson_appropriate_chunks, general_chunks, current_lesson
        )

        return {
            "selected_chunks": selected_chunks,
            "lesson_appropriate_chunks": lesson_appropriate_chunks,
            "general_chunks": general_chunks,
            "future_lesson_chunks": future_lesson_chunks
        }

    def _get_accessible_lessons(self, current_lesson: str) -> List[int]:
        """Determine which lessons student should have access to"""
        if current_lesson == "all":
            return None  # All lessons available
        else:
            try:
                current_lesson_num = int(current_lesson)
                return list(range(1, current_lesson_num + 1))  # Lessons 1 through current
            except (ValueError, TypeError):
                return None  # Fallback to all lessons

    def _extract_chunk_info(self, match) -> Dict:
        """Extract chunk information from search match"""
        return {
            "text": match.metadata.get('text', ''),
            "score": match.score,
            "lesson": match.metadata.get('lesson_number', ''),
            "hierarchy_path": match.metadata.get('hierarchy_path', ''),
            "document": match.metadata.get('document_name', ''),
            "chunk_type": match.metadata.get('chunk_type', 'general')
        }

    def _calculate_chunk_score(self, chunk: Dict, current_lesson: str,
                             conversation_context: ConversationContext) -> float:
        """Calculate enhanced scoring for chunk ranking"""
        score = chunk['score']

        # Lesson-based scoring
        if current_lesson != "all" and chunk['lesson']:
            try:
                lesson_num = int(chunk['lesson'])
                current_num = int(current_lesson)

                # Major boost for current lesson content (90% priority)
                if lesson_num == current_num:
                    score += 1.5

                # Strategic connection boosts for learning flow (10% priority)
                elif lesson_num < current_num:
                    lessons_back = current_num - lesson_num
                    if lessons_back == 1:
                        score += 0.2
                    elif lessons_back == 2:
                        score += 0.15
                    elif lessons_back <= 4:
                        score += 0.08
                    elif lessons_back <= 6:
                        score += 0.03
                    else:
                        score += 0.01

                # No boost for future lessons
                elif lesson_num > current_num:
                    score -= 1.0

            except (ValueError, TypeError):
                pass

        # Current lesson academic structure priority
        if current_lesson != "all" and chunk['lesson'] == current_lesson:
            if chunk['chunk_type'] == 'lesson_content':
                score += 0.2
            if chunk.get('hierarchy_path', '').startswith(f"Lesson {current_lesson}"):
                score += 0.15

        # General academic structure boosts
        if chunk['chunk_type'] == 'lesson_content':
            score += 0.05
        if chunk['lesson']:
            score += 0.03

        # Conversation continuity boost
        chunk_topic = chunk.get('hierarchy_path', '').lower()
        for discussed_topic in conversation_context.topics_discussed:
            if discussed_topic.lower() in chunk_topic or chunk_topic in discussed_topic.lower():
                score += 0.08
                break

        return score

    def _select_final_chunks(self, lesson_appropriate_chunks: List[Dict],
                           general_chunks: List[Dict], current_lesson: str) -> List[Dict]:
        """Select final chunks with lesson focus and strategic connections"""
        selected_chunks = []

        if current_lesson != "all":
            # For specific lesson: prioritize current lesson heavily
            current_lesson_chunks = [chunk for chunk in lesson_appropriate_chunks
                                   if chunk.get('lesson') == current_lesson]
            connection_chunks = [chunk for chunk in lesson_appropriate_chunks
                               if chunk.get('lesson') != current_lesson and chunk.get('lesson')]

            # Take 2-3 chunks from current lesson (90% priority)
            selected_chunks.extend(current_lesson_chunks[:3])

            # Add 1 connection chunk if available and space remains (10% priority)
            if len(selected_chunks) < 3 and connection_chunks:
                selected_chunks.extend(connection_chunks[:1])

            # Fill any remaining slots with general content
            remaining_slots = max(0, 3 - len(selected_chunks))
            if remaining_slots > 0:
                selected_chunks.extend(general_chunks[:remaining_slots])
        else:
            # For "all lessons": get comprehensive coverage of the topic
            # Take more chunks to get complete picture across lessons
            selected_chunks.extend(lesson_appropriate_chunks[:5])
            remaining_slots = max(0, 6 - len(selected_chunks))
            if remaining_slots > 0:
                selected_chunks.extend(general_chunks[:remaining_slots])

        return selected_chunks

    def _format_context_for_response(self, selected_chunks: List[Dict]) -> str:
        """Format context with academic structure for response generation"""
        context_parts = []
        for i, chunk in enumerate(selected_chunks):
            context_part = f"=== COURSE MATERIAL {i+1} ==="
            if chunk['lesson']:
                context_part += f" (Lesson {chunk['lesson']})"
            if chunk['hierarchy_path'] and chunk['hierarchy_path'] != "General Content":
                context_part += f"\nTopic: {chunk['hierarchy_path']}"
            context_part += f"\nSource: {chunk['document']}"
            context_part += f"\n\n{chunk['text']}"
            context_parts.append(context_part)

        return "\n\n" + "="*60 + "\n\n".join(context_parts) + "\n\n" + "="*60

    def sanitize_context_metadata(self, context: str) -> str:
        """Clean up metadata while preserving content structure and readability"""
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

        # Clean up leading/trailing whitespace and section markers
        sanitized = sanitized.strip()

        # Add proper spacing around section breaks
        sanitized = re.sub(r'--- SECTION ---', '\n\n--- SECTION ---\n', sanitized)

        return sanitized.strip()

    def get_alternative_content_from_embeddings(self, current_concept: str,
                                              current_chunks: List[Dict],
                                              course_namespace: str) -> List[Dict]:
        """Get alternative content related to the concept from vector embeddings"""
        try:
            if not current_chunks:
                return []

            # Get the embedding for the current concept
            concept_embedding = self.embedding_model.encode([current_concept])[0].tolist()

            # Search for related but different content
            related_results = self.pinecone_index.query(
                vector=concept_embedding,
                top_k=15,  # Get more results to have variety
                include_metadata=True,
                namespace=course_namespace,
                filter={"course": course_namespace}
            )

            # Filter out chunks we've already used and extract diverse examples
            current_chunk_ids = {chunk.get('id', '') for chunk in current_chunks}
            alternative_chunks = []

            for match in related_results.matches:
                chunk_id = match.get('id', '')
                if chunk_id not in current_chunk_ids and match.score > 0.5:
                    chunk_info = {
                        "text": match.metadata.get('text', ''),
                        "hierarchy_path": match.metadata.get('hierarchy_path', ''),
                        "document": match.metadata.get('document_name', ''),
                        "lesson": match.metadata.get('lesson_number', ''),
                        "score": match.score
                    }
                    alternative_chunks.append(chunk_info)

            return alternative_chunks[:5]  # Top 5 alternative chunks

        except Exception as e:
            logger.error(f"Error getting alternative content from embeddings: {e}")
            return []

    def analyze_lesson_content_structure(self, chunks: List[Dict], current_lesson: str) -> Dict:
        """Analyze the lesson content structure of search results"""
        current_lesson_chunks = []
        connection_chunks = []
        general_chunks = []

        for chunk in chunks:
            lesson = chunk.get('lesson')
            if lesson:
                if str(lesson) == str(current_lesson):
                    current_lesson_chunks.append(chunk)
                else:
                    connection_chunks.append(chunk)
            else:
                general_chunks.append(chunk)

        return {
            "current_lesson_chunks": current_lesson_chunks,
            "connection_chunks": connection_chunks,
            "general_chunks": general_chunks,
            "lesson_distribution": {
                "current": len(current_lesson_chunks),
                "connections": len(connection_chunks),
                "general": len(general_chunks)
            }
        }