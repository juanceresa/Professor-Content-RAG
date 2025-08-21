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

def get_available_courses_from_pinecone(pinecone_index):
    """Dynamically discover courses from Pinecone namespaces"""
    try:
        # Get index stats to see available namespaces
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.get('namespaces', {})

        courses = {}
        for namespace in namespaces.keys():
            if namespace:  # Skip empty namespace
                # Convert namespace back to display name
                display_name = namespace.replace("_", " ").title()

                courses[namespace] = {
                    "name": display_name,
                    "namespace": namespace,
                    "description": f"Course materials for {display_name}",
                    "system_prompt": f"""You are Professor [Father's Name], teaching {display_name}.
                    Answer questions based on the course materials provided. Be scholarly but accessible to undergraduate students.
                    If you don't have specific information from the course materials, say so clearly.""",
                    "vector_count": namespaces[namespace].get('vector_count', 0)
                }

        return courses
    except Exception as e:
        logger.error(f"Error loading courses from Pinecone: {e}")
        # Fallback to default courses if Pinecone unavailable
        return {
            "comparative_politics": {
                "name": "Comparative Politics",
                "namespace": "comparative_politics",
                "description": "Comparing political systems and institutions",
                "system_prompt": "You are a political science professor specializing in comparative politics."
            },
            "international_relations": {
                "name": "International Relations",
                "namespace": "international_relations",
                "description": "Global politics and international affairs",
                "system_prompt": "You are a political science professor specializing in international relations."
            }
        }

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

def search_course_content(query: str, course_key: str, services: Dict, top_k: int = 5) -> str:
    """Search for relevant content in the specific course namespace"""
    try:
        # Generate query embedding
        query_embedding = services["embedding_model"].encode([query])[0].tolist()

        # Get course namespace
        namespace = COURSES[course_key]["namespace"]

        # Search Pinecone
        results = services["pinecone"].query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter={"course": namespace}
        )

        if not results.matches:
            return "No relevant course material found for this query."

        # Combine relevant chunks
        context_chunks = []
        for match in results.matches:
            if match.score > 0.7:  # Only use high-confidence matches
                context_chunks.append(match.metadata.get('text', ''))

        if not context_chunks:
            return "No sufficiently relevant course material found for this query."

        return "\n\n".join(context_chunks[:3])  # Limit to top 3 chunks

    except Exception as e:
        logger.error(f"Error searching course content: {e}")
        return f"Error retrieving course content: {str(e)}"

def generate_response(prompt: str, context: str, course_key: str, services: Dict) -> str:
    """Generate response using Google AI with course-specific context"""
    try:
        system_prompt = COURSES[course_key]["system_prompt"]

        full_prompt = f"""{system_prompt}

COURSE CONTEXT:
{context}

STUDENT QUESTION: {prompt}

Please provide a helpful response based on the course materials above. If the context doesn't contain relevant information, please say so clearly."""

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
                context = search_course_content(prompt, selected_course_key, services)

                # Generate response
                response = generate_response(prompt, context, selected_course_key, services)

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