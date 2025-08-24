# AI Professor Platform

A RAG-based chatbot system that creates course-specific AI assistants for educational institutions. The platform provides perfect course isolation, allowing students to interact with AI that knows only their specific class content.

## Overview

The AI Professor Platform enables educators to upload course materials and create intelligent, course-specific chatbots. Students can then engage with AI assistants that have been trained exclusively on their course content, preventing cross-contamination between different classes and ensuring focused, relevant responses.

## Key Features

- **Perfect Course Isolation**: Each course operates in its own isolated namespace
- **Multi-format Document Support**: Processes PDFs, Word documents (.docx, .doc)
- **Hierarchical Content Processing**: Preserves academic structure (lessons → topics → subtopics)
- **Intelligent Chunking**: Maintains semantic boundaries while creating searchable content chunks
- **Context-Aware Responses**: Generates responses using only relevant course materials
- **Cost-Effective Architecture**: Built on free tiers for MVP validation

## Architecture

### Tech Stack
- **Frontend**: Streamlit web interface
- **Vector Database**: Pinecone (with course-specific namespaces)
- **LLM**: Google AI/Gemini for response generation
- **Embeddings**: Local sentence-transformers model
- **Document Processing**: PyMuPDF and python-docx for text extraction

### Core Components

1. **Document Processing Pipeline** (`embed_docs.py`)
   - Extracts text from academic documents
   - Creates hierarchical chunks preserving lesson structure
   - Generates embeddings using local models
   - Uploads to course-specific vector database namespaces

2. **Course Management System** (`course_manager.py`)
   - Manages course creation and configuration
   - Handles course-specific settings and metadata
   - Dynamically discovers available courses

3. **Content Search Engine** (`content_search.py`)
   - Performs semantic search within course boundaries
   - Retrieves relevant content chunks for student queries
   - Filters results by confidence and relevance

4. **Pedagogical Engine** (`pedagogical_engine.py`)
   - Applies educational best practices to responses
   - Adapts communication style for learning contexts
   - Maintains academic tone and structure

5. **Response Generator** (`response_generator.py`)
   - Combines retrieved content with student questions
   - Generates contextual, course-specific responses
   - Ensures responses stay within course boundaries

6. **Conversation Manager** (`conversation_manager.py`)
   - Maintains conversation history and context
   - Tracks student progress and engagement
   - Manages session state and continuity

## Installation

### Requirements
- Python 3.8+
- Pinecone account (free tier)
- Google AI Studio API key (free tier)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd ai-professor

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create .streamlit/secrets.toml with:
# GOOGLE_AI_API_KEY = "your-google-ai-api-key"
# PINECONE_API_KEY = "your-pinecone-api-key"  
# PINECONE_INDEX_NAME = "your-index-name"
```

## Usage

### Processing Course Documents
```bash
# Process all course materials in documents/ directory
python embed_docs.py

# Or use the processing runner
python run_processing.py
```

### Running the Application
```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Or use the OOP version
streamlit run streamlit_app_oop.py
```

### Document Organization
Place course materials in the following structure:
```
documents/
├── course-name-1/
│   ├── lecture1.pdf
│   ├── chapter1.docx
│   └── readings.doc
├── course-name-2/
│   ├── materials.pdf
│   └── syllabus.docx
└── ...
```

## Testing

The platform includes comprehensive testing utilities:

```bash
# Test document processing
python tests/test_chunking.py

# Test content retrieval
python tests/test_enhanced_chunking.py

# Test pedagogical features
python tests/test_pedagogical_features.py

# Debug processing pipeline
python tests/debug_processing.py
```

## Features in Development

- User authentication and course enrollment
- Advanced analytics and usage tracking
- Multi-modal content support (images, videos)
- Integration with learning management systems
- Administrative dashboard for educators

## Technical Approach

### Course Isolation
Each course gets its own Pinecone namespace, ensuring complete separation of content. Students can only access materials from courses they're enrolled in.

### Hierarchical Processing
The system recognizes academic document structures (bullet points, numbered lists, headings) and preserves these relationships during chunking, improving search relevance and response quality.

### Semantic Search
Uses sentence transformers to create embeddings that capture semantic meaning, allowing students to ask questions in natural language and receive relevant answers even when exact keywords don't match.

### Educational Optimization
The pedagogical engine ensures responses are formatted for learning, with appropriate academic tone, clear explanations, and structured information presentation.

## Contributing

This is an educational platform designed to enhance learning through AI assistance. Contributions should focus on improving educational outcomes, accessibility, and platform reliability.

## License

Educational use license - designed for academic institutions and learning environments.
