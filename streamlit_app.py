import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Professor Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None
if "selected_lesson" not in st.session_state:
    st.session_state.selected_lesson = "all"  # Default to all lessons
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Enhanced conversation context for pedagogical teaching
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "topics_discussed": [],
        "concepts_introduced": [],
        "student_understanding_signals": [],
        "question_types": [],  # track if asking for definitions, applications, analysis, etc.
        "learning_progression": [],
        "last_topic": None,
        "conversation_depth": "surface",  # surface, intermediate, deep
        "teaching_mode": "guided_discovery",  # guided_discovery, socratic, direct_instruction
        "hint_level": 0,  # progressive hint system
        "session_start_time": None,
        # New dynamic conversation features
        "examples_used": [],  # track which examples have been given
        "content_angles_explored": [],  # track which aspects of topics covered
        "student_engagement_level": "neutral",  # engaged, neutral, disengaged
        "consecutive_minimal_responses": 0,  # track short/low-engagement responses
        "concept_repetition_count": {},  # count how many times each concept explained
        "preferred_learning_style": "unknown",  # concrete, theoretical, application-focused
        "conversation_branch_history": [],  # track which content branches taken
        # Lesson-based learning progression
        "current_lesson": "all",  # lesson student is currently on
        "available_lessons": [],  # lessons available in this course
        "lesson_progression_history": [],  # track lesson changes during session
    }

# Course configurations - will be dynamically loaded from Pinecone namespaces
COURSES = {}

def analyze_student_engagement(response: str) -> str:
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

def detect_content_repetition(topic: str, concept: str) -> bool:
    """Check if we've covered this concept too many times recently"""
    context = st.session_state.conversation_context
    
    # Check if this specific concept has been repeated
    if concept in context["concept_repetition_count"]:
        if context["concept_repetition_count"][concept] >= 2:
            return True
    
    # Check if we're stuck on the same topic
    recent_topics = context["topics_discussed"][-3:]
    if recent_topics.count(topic) >= 2:
        return True
    
    return False

def get_available_lessons_for_course(pinecone_index, course_namespace: str) -> list:
    """Get available lessons for a course by scanning the vector index metadata"""
    try:
        # Query for a broad sample of the course content to find lesson numbers
        sample_results = pinecone_index.query(
            vector=[0.0] * 768,  # Zero vector to get diverse results (768d for all-mpnet-base-v2)
            top_k=100,  # Get many results to find all lessons
            include_metadata=True,
            namespace=course_namespace,
            filter={"course": course_namespace}
        )
        
        # Extract unique lesson numbers
        lessons = set()
        for match in sample_results.matches:
            lesson_num = match.metadata.get('lesson_number', '')
            if lesson_num and lesson_num.isdigit():
                lessons.add(int(lesson_num))
        
        # Return sorted list of available lessons
        sorted_lessons = sorted(list(lessons))
        logger.info(f"Found lessons {sorted_lessons} for course {course_namespace}")
        return sorted_lessons
        
    except Exception as e:
        logger.error(f"Error getting available lessons: {e}")
        return []

def get_alternative_content_from_embeddings(current_concept: str, current_chunks: list, services: Dict, course_key: str) -> list:
    """Get alternative content related to the concept from vector embeddings"""
    try:
        if not current_chunks:
            return []
        
        # Get the embedding for the current concept
        concept_embedding = services["embedding_model"].encode([current_concept])[0].tolist()
        
        # Get course namespace
        namespace = COURSES[course_key]["namespace"]
        
        # Search for related but different content
        related_results = services["pinecone"].query(
            vector=concept_embedding,
            top_k=15,  # Get more results to have variety
            include_metadata=True,
            namespace=namespace,
            filter={"course": namespace}
        )
        
        # Filter out chunks we've already used and extract diverse examples
        current_chunk_ids = {chunk.get('id', '') for chunk in current_chunks}
        alternative_chunks = []
        
        for match in related_results.matches:
            chunk_id = match.get('id', '')
            if chunk_id not in current_chunk_ids and match.score > 0.5:  # Related but different
                chunk_info = {
                    "text": match.metadata.get('text', ''),
                    "hierarchy_path": match.metadata.get('hierarchy_path', ''),
                    "document": match.metadata.get('document_name', ''),
                    "lesson": match.metadata.get('lesson_number', ''),
                    "score": match.score
                }
                alternative_chunks.append(chunk_info)
        
        # Return diverse alternative content
        return alternative_chunks[:5]  # Top 5 alternative chunks
        
    except Exception as e:
        logger.error(f"Error getting alternative content from embeddings: {e}")
        return []

def analyze_question_type(question: str) -> str:
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

def update_conversation_context(question: str, search_results: Dict):
    """Update conversation context based on current interaction"""
    import time
    
    # Initialize session start time if not set
    if st.session_state.conversation_context["session_start_time"] is None:
        st.session_state.conversation_context["session_start_time"] = time.time()
    
    # Analyze student engagement if this is a follow-up (check if there are previous messages)
    if len(st.session_state.messages) > 0:
        engagement = analyze_student_engagement(question)
        st.session_state.conversation_context["student_engagement_level"] = engagement
        
        # Track consecutive minimal responses
        if engagement in ["disengaged", "minimal"]:
            st.session_state.conversation_context["consecutive_minimal_responses"] += 1
        else:
            st.session_state.conversation_context["consecutive_minimal_responses"] = 0
    
    # Analyze and store question type
    question_type = analyze_question_type(question)
    st.session_state.conversation_context["question_types"].append(question_type)
    
    # Extract topics from search results and track concept repetition
    if search_results.get("chunks"):
        for chunk in search_results["chunks"]:
            topic = chunk.get("hierarchy_path", "")
            lesson = chunk.get("lesson", "")
            
            if topic:
                # Track concept repetition
                concept_key = topic.lower()
                if concept_key in st.session_state.conversation_context["concept_repetition_count"]:
                    st.session_state.conversation_context["concept_repetition_count"][concept_key] += 1
                else:
                    st.session_state.conversation_context["concept_repetition_count"][concept_key] = 1
                
                # Add to topics discussed if not already there
                if topic not in st.session_state.conversation_context["topics_discussed"]:
                    st.session_state.conversation_context["topics_discussed"].append(topic)
            
            if lesson:
                st.session_state.conversation_context["last_topic"] = f"Lesson {lesson}: {topic}"
    
    # Adjust conversation depth based on question types
    recent_questions = st.session_state.conversation_context["question_types"][-3:]
    if len(recent_questions) >= 2:
        if all(q in ["analysis", "synthesis"] for q in recent_questions[-2:]):
            st.session_state.conversation_context["conversation_depth"] = "deep"
        elif any(q in ["application", "analysis"] for q in recent_questions):
            st.session_state.conversation_context["conversation_depth"] = "intermediate"

def get_pedagogical_prompt_strategy(question_type: str, conversation_context: Dict) -> str:
    """Generate balanced teaching strategy that informs first, then encourages exploration"""
    depth = conversation_context["conversation_depth"]
    engagement = conversation_context["student_engagement_level"]
    consecutive_minimal = conversation_context["consecutive_minimal_responses"]
    
    # Handle disengagement with different strategies
    if engagement == "disengaged" or consecutive_minimal >= 2:
        return """STUDENT APPEARS DISENGAGED - CHANGE APPROACH:
        - Use a completely different angle or example than previous responses
        - Start with something concrete and relatable (current events, personal experience)
        - Simplify the explanation and use more engaging, conversational tone
        - Ask a different type of question - perhaps more personal or practical
        - Consider switching to a related but different concept from course materials
        - Use vivid, specific examples rather than abstract theory"""
    
    strategies = {
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
    
    return strategies.get(question_type, {}).get(depth, "Provide substantive information from course materials with clear explanations and examples. End with one thoughtful question that encourages deeper exploration.")

def get_course_mapping():
    """Get the course mapping configuration"""
    return {
        "federal_state_local": {
            "name": "Federal, State, and Local Government",
            "namespace": "federal_state_local",
            "description": "American government systems at federal, state, and local levels",
            "system_prompt": """You are Professor Robert Ceresa, teaching American Government with a focus on pedagogical excellence.
            You specialize in federal, state, and local government systems, institutions, and processes.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover answers rather than giving direct answers
            - Use Socratic questioning to develop critical thinking
            - Build on prior knowledge and previously discussed concepts
            - Encourage deeper analysis and connections between ideas
            - Adapt your teaching style based on the student's demonstrated understanding level
            
            RESPONSE APPROACH:
            - When students ask definitional questions, first ask what they already know
            - For application questions, present scenarios and guide them through the reasoning process
            - For analysis questions, break complex topics into manageable components
            - Always reference course materials but encourage students to think critically about the content
            - If you don't have specific information from the course materials, guide them to think about related concepts they do know"""
        },
        "american": {
            "name": "American Political System",
            "namespace": "american",
            "description": "American political institutions, processes, and governance",
            "system_prompt": """You are Professor Robert Ceresa, teaching American Politics with a focus on pedagogical excellence.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover answers rather than giving direct answers
            - Use Socratic questioning to develop critical thinking
            - Build on prior knowledge and previously discussed concepts
            - Encourage deeper analysis and connections between ideas
            - Adapt your teaching style based on the student's demonstrated understanding level
            
            RESPONSE APPROACH:
            - When students ask definitional questions, first ask what they already know
            - For application questions, present scenarios and guide them through the reasoning process
            - For analysis questions, break complex topics into manageable components
            - Always reference course materials but encourage students to think critically about the content
            - If you don't have specific information from the course materials, guide them to think about related concepts they do know"""
        },
        "foundational": {
            "name": "Foundational Political Theory",
            "namespace": "foundational",
            "description": "Core concepts and foundations of political science",
            "system_prompt": """You are Professor Robert Ceresa, teaching Foundational Political Theory with a focus on pedagogical excellence.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover answers rather than giving direct answers
            - Use Socratic questioning to develop critical thinking
            - Build on prior knowledge and previously discussed concepts
            - Encourage deeper analysis and connections between ideas
            - Help students connect foundational theories to contemporary applications
            
            RESPONSE APPROACH:
            - When students ask definitional questions, first ask what they already know
            - For theoretical questions, guide them to examine underlying assumptions and implications
            - For analysis questions, break complex topics into manageable components
            - Always reference course materials but encourage students to think critically about the content
            - Connect classical theories to modern political phenomena where appropriate"""
        },
        "functional": {
            "name": "Functional Political Analysis",
            "namespace": "functional",
            "description": "Functional approaches to understanding political systems",
            "system_prompt": """You are Professor Robert Ceresa, teaching Functional Political Analysis with a focus on pedagogical excellence.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover analytical frameworks rather than giving direct answers
            - Use Socratic questioning to develop systems thinking and functional analysis skills
            - Build on prior knowledge and previously discussed analytical concepts
            - Encourage deeper analysis of how political systems function in practice
            - Help students apply functional analysis to real-world political phenomena
            
            RESPONSE APPROACH:
            - When students ask about functional concepts, first explore their understanding of systems thinking
            - For analytical questions, guide them through the functional analysis process step by step
            - For application questions, present case studies and guide them through functional interpretation
            - Always reference course materials but encourage students to think analytically about political functions
            - Help students see connections between different functional aspects of political systems"""
        },
        "international": {
            "name": "International Relations & Comparative Politics",
            "namespace": "international",
            "description": "International relations, comparative politics, and global affairs",
            "system_prompt": """You are Professor Robert Ceresa, teaching International Relations and Comparative Politics with a focus on pedagogical excellence.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover patterns in international relations rather than giving direct answers
            - Use Socratic questioning to develop comparative analytical thinking
            - Build on prior knowledge and previously discussed international concepts
            - Encourage deeper analysis of global political phenomena and cross-national comparisons
            - Help students understand complex international dynamics through guided exploration
            
            RESPONSE APPROACH:
            - When students ask about international concepts, first explore what they know about global politics
            - For comparative questions, guide them through systematic comparison methodologies
            - For theoretical questions, help them apply IR theories to contemporary global events
            - Always reference course materials but encourage students to think critically about international relations
            - Help students see connections between domestic politics and international outcomes"""
        },
        "professional": {
            "name": "Professional & Management Politics",
            "namespace": "professional",
            "description": "Professional development and management in political contexts",
            "system_prompt": """You are Professor Robert Ceresa, teaching Professional and Management Politics with a focus on pedagogical excellence.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover professional political skills rather than giving direct answers
            - Use Socratic questioning to develop practical political management abilities
            - Build on prior knowledge and previously discussed professional concepts
            - Encourage deeper analysis of political management and professional development
            - Help students apply theoretical knowledge to practical political scenarios
            
            RESPONSE APPROACH:
            - When students ask about professional concepts, first explore their practical experience
            - For management questions, guide them through decision-making processes step by step
            - For application questions, present professional scenarios and guide problem-solving
            - Always reference course materials but encourage students to think practically about political careers
            - Help students connect academic theory to professional political practice"""
        },
        "theory": {
            "name": "Political Philosophy & Theory",
            "namespace": "theory",
            "description": "Classical and modern political philosophy and theory",
            "system_prompt": """You are Professor Robert Ceresa, teaching Political Philosophy and Theory with a focus on pedagogical excellence.
            You specialize in classical and contemporary political thought, from Aristotle and Plato to modern theorists.
            
            TEACHING PHILOSOPHY:
            - Guide students to discover philosophical insights rather than giving direct answers
            - Use Socratic questioning to develop critical thinking about fundamental political questions
            - Build on prior knowledge and previously discussed philosophical concepts
            - Encourage deeper analysis and connections between different philosophical traditions
            - Help students develop their own reasoned positions on political questions
            
            RESPONSE APPROACH:
            - When students ask about philosophical concepts, first explore what they already understand
            - For theoretical questions, guide them to examine underlying assumptions and implications
            - For comparison questions, help them identify key similarities and differences through guided discovery
            - Always reference course materials but encourage students to think critically and philosophically
            - Connect classical theories to contemporary political and ethical dilemmas where appropriate"""
        }
    }

def get_available_courses_from_pinecone(pinecone_index):
    """Get courses that exist in Pinecone based on our course mapping"""
    try:
        # Get index stats to see available namespaces
        stats = pinecone_index.describe_index_stats()
        existing_namespaces = set(stats.get('namespaces', {}).keys())

        # Get our predefined course mapping
        course_mapping = get_course_mapping()

        # Only return courses that exist in Pinecone
        available_courses = {}
        for course_key, course_config in course_mapping.items():
            namespace = course_config['namespace']
            if namespace in existing_namespaces:
                # Add vector count information
                vector_count = stats['namespaces'][namespace].get('vector_count', 0)
                course_config['vector_count'] = vector_count
                available_courses[course_key] = course_config
                logger.info(f"Found course: {course_config['name']} ({vector_count} vectors)")

        return available_courses

    except Exception as e:
        logger.error(f"Error loading courses from Pinecone: {e}")
        # Return empty dict on error - will trigger error message in app
        return {}

@st.cache_resource
def initialize_services():
    """Initialize all external services"""
    try:
        # Initialize Google AI
        genai.configure(api_key=st.secrets["GOOGLE_AI_API_KEY"])

        # Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])

        # Initialize embedding model
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        return {
            "pinecone": index,
            "embedding_model": embedding_model,
            "genai_model": genai.GenerativeModel('gemini-1.5-flash')
        }
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None

def search_course_content(query: str, course_key: str, services: Dict, top_k: int = 8) -> Dict:
    """Search for relevant content with lesson-aware filtering and enhanced academic structure preservation"""
    try:
        # Enhanced query with conversation context
        context_enhanced_query = query
        
        # Add previously discussed topics to enhance search relevance
        if st.session_state.conversation_context["topics_discussed"]:
            recent_topics = st.session_state.conversation_context["topics_discussed"][-2:]
            if recent_topics:
                context_enhanced_query = f"{query} {' '.join(recent_topics)}"
        
        # Generate query embedding with context enhancement
        query_embedding = services["embedding_model"].encode([context_enhanced_query])[0].tolist()

        # Get course namespace
        namespace = COURSES[course_key]["namespace"]
        
        # Get current lesson selection
        current_lesson = st.session_state.selected_lesson
        
        # Build lesson-aware filter
        base_filter = {"course": namespace}
        
        # If specific lesson selected, search more broadly but we'll prioritize later
        if current_lesson != "all":
            # Don't filter at query level - get broader results and prioritize during ranking
            search_top_k = top_k * 3  # Get more results for better lesson-based filtering
        else:
            search_top_k = top_k

        # Search Pinecone with lesson considerations
        results = services["pinecone"].query(
            vector=query_embedding,
            top_k=search_top_k,
            include_metadata=True,
            namespace=namespace,
            filter=base_filter
        )

        if not results.matches:
            return {"context": "No relevant course material found for this query.", "chunks": []}

        # Group and rank chunks by relevance, academic structure, AND lesson progression
        lesson_appropriate_chunks = []  # Chunks that fit current lesson scope
        future_lesson_chunks = []  # Chunks from future lessons (to exclude)
        general_chunks = []  # Chunks without lesson numbers
        
        # Determine which lessons student should have access to
        if current_lesson == "all":
            accessible_lessons = None  # All lessons available
        else:
            try:
                current_lesson_num = int(current_lesson)
                accessible_lessons = list(range(1, current_lesson_num + 1))  # Lessons 1 through current
            except (ValueError, TypeError):
                accessible_lessons = None  # Fallback to all lessons
        
        for match in results.matches:
            chunk_lesson = match.metadata.get('lesson_number', '')
            chunk_info = {
                "text": match.metadata.get('text', ''),
                "score": match.score,
                "lesson": chunk_lesson,
                "hierarchy_path": match.metadata.get('hierarchy_path', ''),
                "document": match.metadata.get('document_name', ''),
                "chunk_type": match.metadata.get('chunk_type', 'general')
            }
            
            # Apply lesson-based filtering
            if not chunk_lesson:
                # General content without lesson number
                if match.score > 0.45:  # Only include if reasonably relevant
                    general_chunks.append(chunk_info)
            elif accessible_lessons is None:
                # All lessons mode
                if match.score > 0.45:
                    lesson_appropriate_chunks.append(chunk_info)
            else:
                # Specific lesson mode - check if lesson is accessible
                try:
                    lesson_num = int(chunk_lesson)
                    if lesson_num in accessible_lessons:
                        lesson_appropriate_chunks.append(chunk_info)
                    else:
                        future_lesson_chunks.append(chunk_info)  # Don't include future lessons
                except (ValueError, TypeError):
                    # Invalid lesson number, treat as general content
                    if match.score > 0.45:
                        general_chunks.append(chunk_info)

        # Enhanced lesson-aware prioritization (90% current lesson, 10% strategic connections)
        def sort_by_lesson_and_academic_value(chunk):
            score = chunk['score']
            
            if current_lesson != "all" and chunk['lesson']:
                try:
                    lesson_num = int(chunk['lesson'])
                    current_num = int(current_lesson)
                    
                    # MAJOR boost for current lesson content (90% priority)
                    if lesson_num == current_num:
                        score += 1.5  # Massive boost for current lesson
                    
                    # Strategic connection boosts for learning flow (10% priority)
                    elif lesson_num < current_num:
                        # Foundational lessons for connection context
                        lessons_back = current_num - lesson_num
                        if lessons_back == 1:  # Immediately previous lesson
                            score += 0.2  # Moderate boost for direct predecessor
                        elif lessons_back == 2:  # Two lessons back
                            score += 0.15  # Good for building connections
                        elif lessons_back <= 4:  # Recent foundational lessons
                            score += 0.08  # Small boost for recent context
                        elif lessons_back <= 6:  # Earlier foundational concepts
                            score += 0.03  # Minimal boost for deep foundations
                        else:  # Early foundational lessons
                            score += 0.01  # Very minimal boost for foundational connections
                    
                    # NO boost for future lessons (completely filtered out)
                    elif lesson_num > current_num:
                        score -= 1.0  # Effectively filter out future content
                        
                except (ValueError, TypeError):
                    pass
            
            # Current lesson academic structure gets additional priority
            if current_lesson != "all" and chunk['lesson'] == current_lesson:
                if chunk['chunk_type'] == 'lesson_content':
                    score += 0.2  # Extra boost for current lesson structured content
                if chunk.get('hierarchy_path', '').startswith(f"Lesson {current_lesson}"):
                    score += 0.15  # Boost for clear current lesson hierarchy
            
            # General academic structure boosts (smaller now)
            if chunk['chunk_type'] == 'lesson_content':
                score += 0.05  # Reduced general lesson content boost
            if chunk['lesson']:
                score += 0.03  # Reduced general lesson boost
            
            # Boost content related to previously discussed topics
            chunk_topic = chunk.get('hierarchy_path', '').lower()
            for discussed_topic in st.session_state.conversation_context["topics_discussed"]:
                if discussed_topic.lower() in chunk_topic or chunk_topic in discussed_topic.lower():
                    score += 0.08  # Maintain conversation continuity boost
                    break
            
            return score

        # Sort all appropriate chunks by lesson and academic value
        lesson_appropriate_chunks.sort(key=sort_by_lesson_and_academic_value, reverse=True)
        general_chunks.sort(key=sort_by_lesson_and_academic_value, reverse=True)

        # Enhanced chunk selection with lesson focus and strategic connections
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
            # For "all lessons": use original logic
            selected_chunks.extend(lesson_appropriate_chunks[:3])
            remaining_slots = max(0, 3 - len(selected_chunks))
            if remaining_slots > 0:
                selected_chunks.extend(general_chunks[:remaining_slots])

        if not selected_chunks:
            return {"context": "No sufficiently relevant course material found for this query.", "chunks": []}

        # Format context with academic structure
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

        formatted_context = "\n\n" + "="*60 + "\n\n".join(context_parts) + "\n\n" + "="*60

        return {
            "context": formatted_context,
            "chunks": selected_chunks,
            "total_matches": len(results.matches)
        }

    except Exception as e:
        logger.error(f"Error searching course content: {e}")
        return {"context": f"Error retrieving course content: {str(e)}", "chunks": []}

def sanitize_context_metadata(context: str) -> str:
    """Remove technical metadata and file details from context for student consumption"""
    import re
    
    # Remove technical headers like "=== COURSE MATERIAL 1 ==="
    sanitized = re.sub(r'=== COURSE MATERIAL \d+ ===.*?\n', '', context)
    
    # Remove source file references like "Source: MASTER Group 2..."
    sanitized = re.sub(r'Source:.*?\n', '', sanitized)
    
    # Remove technical topic paths with brackets
    sanitized = re.sub(r'Topic:.*?\n', '', sanitized)
    
    # Remove equals separators
    sanitized = re.sub(r'={3,}', '', sanitized)
    
    # Clean up extra whitespace
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
    
    return sanitized.strip()

def analyze_lesson_content_structure(chunks: List[Dict]) -> Dict:
    """Analyze the lesson content structure of search results"""
    current_lesson_chunks = []
    connection_chunks = []
    general_chunks = []
    
    for chunk in chunks:
        lesson = chunk.get('lesson')
        if lesson:
            # Lesson-specific content
            if str(lesson) == str(st.session_state.get("selected_lesson", "all")):
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

def generate_response(prompt: str, search_results: Dict, course_key: str, services: Dict) -> str:
    """Generate pedagogical response using conversation context and teaching strategies"""
    try:
        system_prompt = COURSES[course_key]["system_prompt"]
        context = search_results["context"]
        
        # Analyze question and update conversation context
        question_type = analyze_question_type(prompt)
        update_conversation_context(prompt, search_results)
        
        # Get pedagogical strategy based on question type and conversation context
        teaching_strategy = get_pedagogical_prompt_strategy(question_type, st.session_state.conversation_context)
        
        # Analyze content structure for lesson focus
        current_chunks = search_results.get("chunks", [])
        lesson_content_structure = analyze_lesson_content_structure(current_chunks)
        
        # Get current lesson selection for focused instruction
        current_lesson = st.session_state.get("selected_lesson", "all")
        
        # Create lesson-focused instruction
        lesson_instruction = ""
        if current_lesson != "all" and lesson_content_structure["current_lesson_chunks"]:
            lesson_instruction = f"""
LESSON FOCUS PRIORITY (CRITICAL):
- The student is currently studying Lesson {current_lesson}
- 90% of your response should focus on Lesson {current_lesson} content
- Only 10% should reference foundational connections from previous lessons when pedagogically valuable
- NEVER reference future lessons or content the student hasn't reached yet
- When making connections, explicitly note they're building on "previously covered concepts"
"""
        
        # Sanitize metadata from context to hide technical details
        sanitized_context = sanitize_context_metadata(context)
        
        # Check for content repetition and get alternative content from embeddings
        main_concept = ""
        if current_chunks:
            main_concept = current_chunks[0].get("hierarchy_path", "").lower()
        
        repetition_guidance = ""
        if main_concept and detect_content_repetition(main_concept, main_concept):
            # Get alternative content from vector embeddings
            alternative_chunks = get_alternative_content_from_embeddings(main_concept, current_chunks, services, course_key)
            
            if alternative_chunks:
                alternative_topics = [chunk["hierarchy_path"] for chunk in alternative_chunks if chunk["hierarchy_path"]]
                alternative_lessons = [f"Lesson {chunk['lesson']}" for chunk in alternative_chunks if chunk["lesson"]]
                
                repetition_guidance = f"""
CONTENT VARIATION REQUIRED - This concept has been covered recently:
- DO NOT repeat the same examples or explanations from previous responses
- EXPLORE DIFFERENT ANGLES: Consider these related topics from course materials: {', '.join(alternative_topics[:3])}
- ALTERNATIVE LESSONS: Draw from {', '.join(alternative_lessons[:2])} for fresh examples
- Use a completely different approach or perspective on this concept
- Connect to different real-world applications or historical contexts"""
        
        # Build conversation history context
        conversation_history = ""
        if len(st.session_state.messages) > 0:
            recent_messages = st.session_state.messages[-4:]  # Last 2 exchanges
            conversation_history = "\n\nRECENT CONVERSATION CONTEXT:\n"
            for msg in recent_messages:
                role = "STUDENT" if msg["role"] == "user" else "PROFESSOR"
                conversation_history += f"{role}: {msg['content'][:200]}...\n"
        
        # Get topics discussed in this session
        topics_context = ""
        if st.session_state.conversation_context["topics_discussed"]:
            topics_context = f"\n\nTOPICS ALREADY DISCUSSED THIS SESSION:\n{', '.join(st.session_state.conversation_context['topics_discussed'][-5:])}"
        
        # Build lesson-aware context information
        current_lesson = st.session_state.selected_lesson
        lesson_context = ""
        if current_lesson != "all":
            available_lessons = st.session_state.conversation_context.get("available_lessons", [])
            if available_lessons:
                try:
                    current_num = int(current_lesson)
                    completed_lessons = [str(i) for i in range(1, current_num)]
                    lesson_context = f"""
LESSON PROGRESSION CONTEXT:
- Student is currently on Lesson {current_lesson}
- Previously completed lessons: {', '.join(completed_lessons) if completed_lessons else 'None (this is their first lesson)'}
- Do NOT reference content from lessons beyond Lesson {current_lesson}
- Build appropriately on prior lessons when relevant
- Focus primarily on Lesson {current_lesson} concepts"""
                except (ValueError, TypeError):
                    pass
        
        # Build pedagogical instruction based on conversation depth and question type
        pedagogical_instruction = f"""
BALANCED TEACHING APPROACH FOR THIS RESPONSE:
- Question Type: {question_type}
- Conversation Depth: {st.session_state.conversation_context['conversation_depth']}
- Student Engagement: {st.session_state.conversation_context['student_engagement_level']}
- Current Lesson: {current_lesson}
- Teaching Strategy: {teaching_strategy}
{lesson_context}
{repetition_guidance}

RESPONSE STRUCTURE (Follow this order):
1. INFORMATIVE CONTENT (70-80% of response):
   - Provide substantial, clear information from course materials
   - Give comprehensive explanations with concrete examples
   - Include relevant context and why this topic matters
   - Connect to course frameworks and theories

2. STRATEGIC ENGAGEMENT (20-30% of response):
   - End with ONE thoughtful question that encourages deeper thinking
   - Invite exploration of applications, implications, or connections
   - Use phrases like "What aspects interest you most?", "How do you think this applies to...?", "What connections do you see to...?"

TEACHING GUIDELINES:
1. INFORM FIRST: Give students substantive content they can digest and learn from
2. Reference specific course materials, lessons, and examples
3. Build on concepts discussed in previous conversation when relevant
4. Use clear, academic language appropriate for university students
5. Provide enough detail for thorough understanding
6. End with strategic questioning to encourage further exploration
7. Avoid excessive questioning - one good follow-up question is sufficient

CONVERSATION CONTINUITY:
- Connect new information to previous discussions naturally
- Reference earlier concepts to build cumulative knowledge
- Adjust detail level based on demonstrated understanding

CRITICAL: Write ONLY your response as Professor Ceresa. Do NOT include any instructional text, placeholders like [mention a specific example], meta-commentary, or references to "this demonstration". Give a complete, polished academic response."""

        full_prompt = f"""{system_prompt}

{lesson_instruction}

{pedagogical_instruction}

COURSE MATERIALS:
{sanitized_context}
{conversation_history}
{topics_context}

CURRENT STUDENT QUESTION: {prompt}

Respond as Professor Robert Ceresa with a complete, polished academic response. Provide substantive information first, then end with one engaging question. Do not include any instructional text, placeholders, or meta-commentary in your response."""

        response = services["genai_model"].generate_content(full_prompt)
        
        # Track content chunks used in this response to avoid repetition
        if current_chunks:
            for chunk in current_chunks:
                chunk_id = chunk.get('id', f"{chunk.get('hierarchy_path', '')}_{chunk.get('lesson', '')}")
                if chunk_id not in st.session_state.conversation_context["examples_used"]:
                    st.session_state.conversation_context["examples_used"].append(chunk_id)
        
        # Track concepts introduced
        if any(word in prompt.lower() for word in ["what is", "define", "explain"]):
            concept = prompt.split()[-1] if len(prompt.split()) > 2 else "concept"
            if concept not in st.session_state.conversation_context["concepts_introduced"]:
                st.session_state.conversation_context["concepts_introduced"].append(concept)
        
        return response.text

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error generating a response: {str(e)}"

def save_feedback(message_index: int, feedback: str):
    """Save feedback for a specific message"""
    if message_index < len(st.session_state.messages):
        st.session_state.messages[message_index]["feedback"] = feedback
        logger.info(f"Feedback saved for message {message_index}: {feedback}")

# Main UI
st.title("🎓 AI Professor Platform")
st.markdown("*Chat with course-specific AI assistants trained on your professor's materials*")

# Initialize services
services = initialize_services()
if not services:
    st.error("Failed to initialize required services. Please check your API keys.")
    st.stop()

# Load courses from Pinecone
COURSES = get_available_courses_from_pinecone(services["pinecone"])
if not COURSES:
    st.error("No courses found in Pinecone. Please process some documents first.")
    st.info("Run `python embed_docs.py` to process course materials.")
    st.stop()

# Sidebar - Course Selection
with st.sidebar:
    st.header("📚 Course Selection")

    course_options = [(key, info["name"]) for key, info in COURSES.items()]
    course_names = [name for _, name in course_options]
    course_keys = [key for key, _ in course_options]

    selected_course_name = st.selectbox(
        "Select your course:",
        course_names,
        index=0 if not st.session_state.selected_course else course_keys.index(st.session_state.selected_course)
    )

    # Get selected course key
    selected_course_key = course_keys[course_names.index(selected_course_name)]

    # Check if course changed
    if st.session_state.selected_course != selected_course_key:
        st.session_state.selected_course = selected_course_key
        st.session_state.selected_lesson = "all"  # Reset lesson selection
        st.session_state.messages = []  # Clear chat history when switching courses
        # Reset conversation context for new course
        st.session_state.conversation_context = {
            "topics_discussed": [],
            "concepts_introduced": [],
            "student_understanding_signals": [],
            "question_types": [],
            "learning_progression": [],
            "last_topic": None,
            "conversation_depth": "surface",
            "teaching_mode": "guided_discovery",
            "hint_level": 0,
            "session_start_time": None,
            # Reset dynamic conversation features
            "examples_used": [],
            "content_angles_explored": [],
            "student_engagement_level": "neutral",
            "consecutive_minimal_responses": 0,
            "concept_repetition_count": {},
            "preferred_learning_style": "unknown",
            "conversation_branch_history": [],
            # Reset lesson-based learning progression
            "current_lesson": "all",
            "available_lessons": [],
            "lesson_progression_history": []
        }
        st.rerun()

    # Course description and lesson selector
    if selected_course_key:
        course_info = COURSES[selected_course_key]
        st.markdown(f"**{course_info['name']}**")
        st.markdown(f"*{course_info['description']}*")
        
        # Get available lessons for this course
        namespace = course_info["namespace"]
        available_lessons = get_available_lessons_for_course(services["pinecone"], namespace)
        
        # Update conversation context with available lessons
        st.session_state.conversation_context["available_lessons"] = available_lessons
        
        # Lesson selector
        st.markdown("---")
        st.markdown("### 📖 Current Lesson")
        
        if available_lessons:
            lesson_options = ["all"] + [str(lesson) for lesson in available_lessons]
            lesson_labels = ["All Lessons"] + [f"Lesson {lesson}" for lesson in available_lessons]
            
            # Find current selection index
            current_index = 0
            if st.session_state.selected_lesson in lesson_options:
                current_index = lesson_options.index(st.session_state.selected_lesson)
            
            selected_lesson = st.selectbox(
                "Select your current lesson:",
                lesson_options,
                format_func=lambda x: lesson_labels[lesson_options.index(x)],
                index=current_index,
                help="Choose your current lesson to get relevant content. Future lessons will be hidden."
            )
            
            # Check if lesson changed
            if st.session_state.selected_lesson != selected_lesson:
                st.session_state.selected_lesson = selected_lesson
                st.session_state.conversation_context["current_lesson"] = selected_lesson
                # Add to progression history
                st.session_state.conversation_context["lesson_progression_history"].append({
                    "lesson": selected_lesson,
                    "timestamp": time.time()
                })
                st.rerun()
        else:
            st.info("No lessons detected in course materials.")
            st.session_state.selected_lesson = "all"

    st.markdown("---")
    st.markdown("### 📊 Learning Progress")
    st.metric("Messages this session", len(st.session_state.messages))
    
    # Show current lesson if specific lesson selected
    if st.session_state.selected_lesson != "all":
        st.markdown(f"**Current Focus:** Lesson {st.session_state.selected_lesson}")
    
    # Show learning depth if advanced
    if st.session_state.conversation_context["conversation_depth"] != "surface":
        st.markdown(f"**Learning Depth:** {st.session_state.conversation_context['conversation_depth'].title()}")
    
    # Show question variety if there are multiple types
    if st.session_state.conversation_context["question_types"]:
        recent_types = list(set(st.session_state.conversation_context["question_types"][-3:]))
        if len(recent_types) > 1:
            st.markdown(f"**Question Types:** {', '.join(recent_types).title()}")

    st.markdown("---")
    st.markdown("### ⚠️ Important Notes")
    st.markdown("""
    - This is an MVP demonstration
    - Responses are based on uploaded course materials
    - Always verify important information with official sources
    - No user authentication in this version
    """)

# Main chat interface
if selected_course_key:
    course_info = COURSES[selected_course_key]

    # Display chat header
    st.markdown(f"### 💬 Chat with Professor [Name] - {course_info['name']}")

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Add feedback for assistant messages
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                col1, col2, col3 = st.columns([1, 1, 8])

                with col1:
                    if st.button("👍", key=f"thumbs_up_{i}", disabled=feedback is not None):
                        save_feedback(i, "thumbs_up")
                        st.rerun()

                with col2:
                    if st.button("👎", key=f"thumbs_down_{i}", disabled=feedback is not None):
                        save_feedback(i, "thumbs_down")
                        st.rerun()

                if feedback:
                    st.caption(f"Feedback: {'👍 Helpful' if feedback == 'thumbs_up' else '👎 Not helpful'}")

    # Chat input
    if prompt := st.chat_input(f"Ask about {course_info['name']}..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching course materials and generating response..."):
                # Search for relevant content
                search_results = search_course_content(prompt, selected_course_key, services)

                # Generate response
                response = generate_response(prompt, search_results, selected_course_key, services)

                # Display response
                st.markdown(response)

                # Add feedback buttons
                col1, col2, col3 = st.columns([1, 1, 8])
                message_index = len(st.session_state.messages)

                with col1:
                    if st.button("👍", key=f"thumbs_up_{message_index}"):
                        save_feedback(message_index, "thumbs_up")
                        st.rerun()

                with col2:
                    if st.button("👎", key=f"thumbs_down_{message_index}"):
                        save_feedback(message_index, "thumbs_down")
                        st.rerun()

        # Add assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please select a course from the sidebar to begin chatting!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>AI Professor Platform MVP1 | Built with Streamlit, Pinecone, and Google AI</small>
</div>
""", unsafe_allow_html=True)