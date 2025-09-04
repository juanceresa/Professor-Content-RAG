"""
AI Professor Platform - Object-Oriented Streamlit Application (revised)

Main application orchestrating all components for the AI Professor Platform.
Includes: feedback fix, robust selectbox, history cap, secrets guard,
response timing, and small caching/structure tweaks.
"""

import os
import time
import logging
import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Import our OOP components
from course_manager import CourseManager
from conversation_manager import ConversationManager
from pedagogical_engine import PedagogicalEngine
from content_search import ContentSearchEngine
from response_generator import ResponseGenerator

# ----------------------------------
# Logging
# ----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------
# Helpers
# ----------------------------------

def _require_secret(name: str) -> str:
    """Fetch a secret or halt the app with a friendly error."""
    try:
        return st.secrets[name]
    except Exception:
        st.error(f"Missing secret: {name}. Please add it to .streamlit/secrets.toml")
        st.stop()

MAX_HISTORY = 40  # cap chat history length to keep prompts snappy


class StreamlitApp:
    """Main Streamlit application orchestrating all AI Professor Platform components"""

    def __init__(self):
        self.course_manager = None
        self.conversation_manager = None
        self.pedagogical_engine = None
        self.content_search = None
        self.response_generator = None
        self.services = None

        # Initialize page configuration
        self._setup_page_config()

        # Initialize session state
        self._initialize_session_state()

        # Initialize services and components
        self._initialize_components()

    def _setup_page_config(self):
        """Set up Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Professor Platform",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "selected_course" not in st.session_state:
            st.session_state.selected_course = None
        if "selected_lesson" not in st.session_state:
            st.session_state.selected_lesson = "all"

    @staticmethod
    @st.cache_resource
    def _initialize_services():
        """Initialize all external services (cached for performance)"""
        try:
            # Initialize Google AI
            genai.configure(api_key=_require_secret("GOOGLE_AI_API_KEY"))

            # Initialize Pinecone
            pc = Pinecone(api_key=_require_secret("PINECONE_API_KEY"))
            index = pc.Index(_require_secret("PINECONE_INDEX_NAME"))

            # Initialize embedding model
            embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

            return {
                "pinecone": index,
                "embedding_model": embedding_model,
                "genai_model": genai.GenerativeModel("gemini-1.5-flash"),
            }
        except Exception as e:
            st.error(f"Failed to initialize services: {e}")
            return None

    def _initialize_components(self):
        """Initialize all application components"""
        # Initialize services
        self.services = self._initialize_services()
        if not self.services:
            st.error("Failed to initialize required services. Please check your API keys.")
            st.stop()

        # Initialize core components
        self.course_manager = CourseManager()
        self.conversation_manager = ConversationManager()
        self.pedagogical_engine = PedagogicalEngine()

        # Initialize content search engine
        self.content_search = ContentSearchEngine(
            embedding_model=self.services["embedding_model"],
            pinecone_index=self.services["pinecone"],
        )

        # Initialize response generator
        self.response_generator = ResponseGenerator(
            content_search=self.content_search,
            pedagogical_engine=self.pedagogical_engine,
            conversation_manager=self.conversation_manager,
            genai_model=self.services["genai_model"],
        )

        # Load available courses from Pinecone
        available_courses = self.course_manager.load_available_courses_from_pinecone(
            self.services["pinecone"]
        )

        if not available_courses:
            st.error("No courses found in Pinecone. Please process some documents first.")
            st.info("Run `python embed_docs.py` to process course materials.")
            st.stop()

    def render_sidebar(self):
        """Render the sidebar with course selection and progress tracking"""
        with st.sidebar:
            st.header("📚 Course Selection")

            # Course selection
            course_options = self.course_manager.get_course_options()  # [(key, name), ...]
            if not course_options:
                st.warning("No courses found. Process documents or check Pinecone index.")
                return

            course_names = [name for _, name in course_options]
            course_keys = [key for key, _ in course_options]

            # Safer index resolution
            try:
                default_idx = (
                    course_keys.index(st.session_state.selected_course)
                    if st.session_state.selected_course
                    else 0
                )
            except ValueError:
                default_idx = 0

            selected_course_name = st.selectbox(
                "Select your course:", course_names, index=default_idx
            )

            # Get selected course key
            selected_course_key = course_keys[course_names.index(selected_course_name)]

            # Handle course change
            if st.session_state.selected_course != selected_course_key:
                self._handle_course_change(selected_course_key)

            # Render course information and lesson selector
            if selected_course_key:
                self._render_course_info(selected_course_key)
                self._render_lesson_selector(selected_course_key)

            # Render learning progress
            self._render_learning_progress()

            # Render important disclaimers
            self._render_disclaimers()

    def _handle_course_change(self, new_course_key: str):
        """Handle course selection change"""
        st.session_state.selected_course = new_course_key
        st.session_state.selected_lesson = "all"
        st.session_state.messages = []

        # Reset conversation context for new course
        self.conversation_manager.reset_context()
        st.rerun()

    def _render_course_info(self, course_key: str):
        """Render course information"""
        course_config = self.course_manager.get_course(course_key)
        if course_config:
            st.markdown(f"**{course_config.name}**")
            st.markdown(f"*{course_config.description}*")

    def _render_lesson_selector(self, course_key: str):
        """Render lesson selection interface"""
        # Get available lessons for this course
        available_lessons = self.course_manager.get_available_lessons_for_course(
            self.services["pinecone"], course_key
        )

        # Update conversation context with available lessons
        self.conversation_manager.set_available_lessons(available_lessons)

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
                help="Choose your current lesson to get relevant content. Future lessons will be hidden.",
            )

            # Handle lesson change
            if st.session_state.selected_lesson != selected_lesson:
                st.session_state.selected_lesson = selected_lesson
                self.conversation_manager.update_lesson_selection(selected_lesson)
                st.rerun()
        else:
            st.info("No lessons detected in course materials.")
            st.session_state.selected_lesson = "all"

    def _render_learning_progress(self):
        """Render learning progress metrics"""
        st.markdown("---")
        st.markdown("### 📊 Learning Progress")
        st.metric("Messages this session", len(st.session_state.messages))

        context_summary = self.conversation_manager.get_context_summary()

        # Show current lesson if specific lesson selected
        if st.session_state.selected_lesson != "all":
            st.markdown(f"**Current Focus:** Lesson {st.session_state.selected_lesson}")

        # Show learning depth if advanced
        if context_summary["conversation_depth"] != "surface":
            st.markdown(f"**Learning Depth:** {context_summary['conversation_depth'].title()}")

        # Show question variety if there are multiple types
        if context_summary["question_types"]:
            if len(context_summary["question_types"]) > 1:
                st.markdown(
                    f"**Question Types:** {', '.join(context_summary['question_types']).title()}"
                )

    def _render_disclaimers(self):
        """Render important disclaimers"""
        st.markdown("---")
        st.markdown("### ⚠️ Important Disclaimers")
        st.markdown(
            """
        - **AI-Generated Content:** This AI assistant is not Professor Ceresa himself. All responses are AI-generated based on course materials and may contain errors or inaccuracies.
        - **No Official Endorsement:** Professor Ceresa does not endorse or take responsibility for any AI-generated responses, opinions, or interpretations.
        - **Verify Information:** Always cross-reference important information with official course materials, textbooks, and consult Professor Ceresa directly for authoritative answers.
        - **Academic Integrity:** Use this tool as a study aid only. Follow your institution's academic integrity policies regarding AI assistance.
        - **Not Professional Advice:** This AI does not provide professional, legal, or official academic advice. Contact Professor Ceresa for official guidance.
        - **Privacy Notice:** Messages are not private and may be logged for educational improvement purposes.
        """
        )

    def render_main_chat(self):
        """Render the main chat interface"""
        if not st.session_state.selected_course:
            st.info("Please select a course from the sidebar to begin chatting!")
            return

        course_config = self.course_manager.get_course(st.session_state.selected_course)
        if not course_config:
            st.error("Selected course not found.")
            return

        # Display chat header
        st.markdown(f"### 💬 Chat with Professor Ceresa - {course_config.name}")

        # Display chat history
        self._render_chat_history()

        # Handle new chat input
        self._handle_chat_input(course_config)

    def _render_chat_history(self):
        """Render existing chat messages with feedback buttons"""
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Add feedback for assistant messages
                if message["role"] == "assistant":
                    self._render_feedback_buttons(message, i)

    def _render_feedback_buttons(self, message: dict, message_index: int):
        """Render feedback buttons for assistant messages"""
        feedback = message.get("feedback", None)
        col1, col2, col3 = st.columns([1, 1, 8])

        with col1:
            if st.button("👍", key=f"thumbs_up_{message_index}", disabled=feedback is not None):
                self._save_feedback(message_index, "thumbs_up")
                st.rerun()

        with col2:
            if st.button("👎", key=f"thumbs_down_{message_index}", disabled=feedback is not None):
                self._save_feedback(message_index, "thumbs_down")
                st.rerun()

        if feedback:
            st.caption(f"Feedback: {'👍 Helpful' if feedback == 'thumbs_up' else '👎 Not helpful'}")

    def _handle_chat_input(self, course_config):
        """Handle new chat input and generate response"""
        if prompt := st.chat_input(f"Ask about {course_config.name}..."):
            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Save user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching course materials and generating response..."):
                    t0 = time.time()
                    try:
                        # Try new signature (selected_lesson)
                        response = self.response_generator.generate_response(
                            prompt=prompt,
                            course_config=course_config,
                            messages=st.session_state.messages,
                            selected_lesson=st.session_state.get("selected_lesson", "all"),
                        )
                    except TypeError:
                        # Fallback for older ResponseGenerator
                        response = self.response_generator.generate_response(
                            prompt=prompt,
                            course_config=course_config,
                            messages=st.session_state.messages,
                        )
                    except Exception as e:
                        logging.exception("Response generation failed")
                        response = f"Sorry, I hit an error generating a response: {e}"

                    dt = time.time() - t0

                    # Append assistant message BEFORE rendering feedback buttons
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Display response + feedback
                    st.markdown(response)
                    self._render_feedback_buttons(st.session_state.messages[-1], len(st.session_state.messages) - 1)
                    st.caption(f"Response time: {dt:.1f}s")

            # Cap history length
            if len(st.session_state.messages) > MAX_HISTORY:
                st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

    def _save_feedback(self, message_index: int, feedback: str):
        """Save feedback for a message"""
        if message_index < len(st.session_state.messages):
            st.session_state.messages[message_index]["feedback"] = feedback
            logger.info(f"Feedback saved for message {message_index}: {feedback}")

    def run(self):
        """Main application entry point"""
        st.title("🎓 AI Professor Platform")
        st.markdown("*Chat with course-specific AI assistants trained on your professor's materials*")
        
        self.render_sidebar()
        self.render_main_chat()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            <small>AI Professor Platform | Built with Streamlit, Pinecone, and Google AI</small>
        </div>
        """, unsafe_allow_html=True)


# Application entry point
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
else:
    # For Streamlit Cloud deployment
    app = StreamlitApp()
    app.run()