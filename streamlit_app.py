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
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Course configurations - will be dynamically loaded from Pinecone namespaces
COURSES = {}

def get_course_mapping():
    """Get the course mapping configuration"""
    return {
        "federal_state_local": {
            "name": "Federal, State, and Local Government",
            "namespace": "federal_state_local",
            "description": "American government systems at federal, state, and local levels",
            "system_prompt": """You are Professor Robert Ceresa, teaching American Government.
            You specialize in federal, state, and local government systems, institutions, and processes.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
        },
        "american": {
            "name": "American Political System",
            "namespace": "american",
            "description": "American political institutions, processes, and governance",
            "system_prompt": """You are Professor Robert Ceresa, teaching American Politics.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
        },
        "foundational": {
            "name": "Foundational Political Theory",
            "namespace": "foundational",
            "description": "Core concepts and foundations of political science",
            "system_prompt": """You are Professor Robert Ceresa, teaching Foundational Political Theory.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
        },
        "functional": {
            "name": "Functional Political Analysis",
            "namespace": "functional",
            "description": "Functional approaches to understanding political systems",
            "system_prompt": """You are Professor Robert Ceresa, teaching Functional Political Analysis.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
        },
        "international": {
            "name": "International Relations & Comparative Politics",
            "namespace": "international",
            "description": "International relations, comparative politics, and global affairs",
            "system_prompt": """You are Professor Robert Ceresa, teaching International Relations and Comparative Politics.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
        },
        "professional": {
            "name": "Professional & Management Politics",
            "namespace": "professional",
            "description": "Professional development and management in political contexts",
            "system_prompt": """You are Professor Robert Ceresa, teaching Professional and Management Politics.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
        },
        "theory": {
            "name": "Political Philosophy & Theory",
            "namespace": "theory",
            "description": "Classical and modern political philosophy and theory",
            "system_prompt": """You are Professor Robert Ceresa, teaching Political Philosophy and Theory.
            You specialize in classical and contemporary political thought, from Aristotle and Plato to modern theorists.
            Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
            If you don't have specific information from the course materials, say so clearly."""
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
    """Search for relevant content with enhanced academic structure preservation"""
    try:
        # Generate query embedding
        query_embedding = services["embedding_model"].encode([query])[0].tolist()

        # Get course namespace
        namespace = COURSES[course_key]["namespace"]

        # Search Pinecone with more results for better selection
        results = services["pinecone"].query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter={"course": namespace}
        )

        if not results.matches:
            return {"context": "No relevant course material found for this query.", "chunks": []}

        # Group and rank chunks by relevance and academic structure
        high_relevance_chunks = []  # Score > 0.65
        medium_relevance_chunks = []  # Score 0.45-0.65
        
        for match in results.matches:
            chunk_info = {
                "text": match.metadata.get('text', ''),
                "score": match.score,
                "lesson": match.metadata.get('lesson_number', ''),
                "hierarchy_path": match.metadata.get('hierarchy_path', ''),
                "document": match.metadata.get('document_name', ''),
                "chunk_type": match.metadata.get('chunk_type', 'general')
            }
            
            if match.score > 0.65:
                high_relevance_chunks.append(chunk_info)
            elif match.score > 0.45:
                medium_relevance_chunks.append(chunk_info)

        # Prioritize lesson content and structured content
        def sort_by_academic_value(chunk):
            score = chunk['score']
            if chunk['chunk_type'] == 'lesson_content':
                score += 0.1  # Boost lesson content
            if chunk['lesson']:
                score += 0.05  # Boost content with lesson numbers
            return score

        high_relevance_chunks.sort(key=sort_by_academic_value, reverse=True)
        medium_relevance_chunks.sort(key=sort_by_academic_value, reverse=True)

        # Select best chunks (prefer high relevance, but include medium if needed)
        selected_chunks = high_relevance_chunks[:2]  # Top 2 high relevance
        if len(selected_chunks) < 2:
            selected_chunks.extend(medium_relevance_chunks[:3-len(selected_chunks)])

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

def generate_response(prompt: str, search_results: Dict, course_key: str, services: Dict) -> str:
    """Generate response using Google AI with enhanced academic fidelity"""
    try:
        system_prompt = COURSES[course_key]["system_prompt"]
        context = search_results["context"]
        chunks = search_results.get("chunks", [])

        # Enhanced academic instruction
        academic_instruction = """
IMPORTANT TEACHING GUIDELINES:
1. Stay strictly within the course materials provided
2. Reference specific lessons, concepts, and hierarchical structure when available
3. Use the exact terminology and frameworks from the course materials
4. If the student asks about something not covered in the materials, clearly state this
5. Maintain the academic rigor and teaching style consistent with university-level political science
6. When possible, reference the source lesson or document for further study
7. Present information in a logical, educational sequence that builds understanding"""

        full_prompt = f"""{system_prompt}

{academic_instruction}

COURSE MATERIALS:
{context}

STUDENT QUESTION: {prompt}

Based on the course materials above, provide a scholarly response that:
- Uses the exact concepts, terminology, and frameworks from the materials
- References specific lessons or topics when relevant
- Maintains academic rigor appropriate for university students
- Clearly indicates if the question goes beyond the provided course content"""

        response = services["genai_model"].generate_content(full_prompt)
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
    course_names = [name for key, name in course_options]
    course_keys = [key for key, name in course_options]

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
        st.session_state.messages = []  # Clear chat history when switching courses
        st.rerun()

    # Course description
    if selected_course_key:
        course_info = COURSES[selected_course_key]
        st.markdown(f"**{course_info['name']}**")
        st.markdown(f"*{course_info['description']}*")

    st.markdown("---")
    st.markdown("### 📊 Usage Stats")
    st.metric("Messages this session", len(st.session_state.messages))

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