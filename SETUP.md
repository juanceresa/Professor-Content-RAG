# AI Professor Platform - Setup Guide

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Your Document Structure
```bash
# Validate your documents directory (no dependencies needed)
python test_structure.py
```

**Expected Output:**
```
✅ Found 8 classes with 39 total documents
Course types detected:
  • American: 1 documents
  • Foundational: 4 documents  
  • Functional: 2 documents
  • Govt: 3 documents
  • International: 6 documents
  • Local: 5 documents
  • Professional: 4 documents
  • Theory: 14 documents
```

### 3. Configure API Keys

Create `.streamlit/secrets.toml`:
```toml
GOOGLE_AI_API_KEY = "your-google-ai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"  
PINECONE_INDEX_NAME = "ai-professor-platform"
```

**Get API Keys:**
- **Google AI:** https://aistudio.google.com/ (45K requests/month FREE)
- **Pinecone:** https://pinecone.io/ (1M vectors FREE)

### 4. Process Your Documents
```bash
# Test document processing first
python test_chunking.py

# Process all documents and upload to Pinecone
python embed_docs.py
```

### 5. Run the Application
```bash
streamlit run streamlit_app.py
```

## Your Current Document Structure

**8 Political Science Courses Ready:**

```
documents/
├── american/          (1 document)
├── foundational/      (4 documents) 
├── functional/        (2 documents)
├── govt/              (3 documents)
├── international/     (6 documents)
├── local/             (5 documents)
├── professional/      (4 documents)
└── theory/            (14 documents - largest collection)
```

**Total: 39 documents (5.5 MB) across 8 courses**

## Troubleshooting

### Common Issues

**❌ "No module named 'docx'"**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**❌ "No courses found in Pinecone"**
- First run `python embed_docs.py` to process documents
- Verify API keys in `.streamlit/secrets.toml`

**❌ "Documents directory not found"**
- Ensure you're in the project root directory
- Run `python test_structure.py` to validate

### Installation Issues

If `pip install -r requirements.txt` fails:

```bash
# Install dependencies individually
pip install streamlit
pip install pinecone-client  
pip install sentence-transformers
pip install google-generativeai
pip install python-docx
pip install PyPDF2
pip install PyMuPDF
pip install numpy pandas
pip install python-dotenv
```

## Next Steps After Setup

1. **Test Locally:** Verify all 8 courses appear in the sidebar
2. **Chat Test:** Ask questions about specific topics (e.g., "What is Aristotle's view on politics?")
3. **Deploy:** Push to GitHub and deploy to Streamlit Cloud
4. **Scale:** Add more courses by placing documents in new folders

## Academic Content Optimization

Your content is well-structured for the AI Professor platform:

**✅ Theoretical Foundation** (Theory: 14 docs)
- Classical political philosophy (Aristotle, Plato)
- Modern political thought (Everson series)
- Contemporary theory (Foucault, post-modernism)

**✅ Practical Application** (Govt, Local, Professional: 12 docs)
- American government systems
- Local politics and administration
- Professional/management aspects

**✅ Comparative Analysis** (International, Functional: 8 docs)
- International relations
- Comparative political systems
- Functional political analysis

**✅ Core Concepts** (American, Foundational: 5 docs)
- American political system
- Foundational political concepts

This structure provides comprehensive coverage for students across all political science areas!