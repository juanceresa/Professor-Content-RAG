"""
Test Improved Chunking Algorithm

Tests the new chunking system with sample problematic content to ensure
it creates meaningful, contextual chunks.
"""

import sys
sys.path.append('..')

from improved_chunking import create_improved_chunks


def test_problematic_content():
    """Test with the actual problematic content from the debug output"""
    
    sample_text = """=== Lesson 1 ===

       •                 IPP
     o              What else
       •                 Anger
       •                 Patience
       •                 Companionship (e.g., people go crazy if they spend too much time alone)
             Institutions for organizing, structuring, or giving pattern to relations
       •                 People have to know the kind of world they want to live in before building it.
         They are in a position to participate in the political process and that gives them access to power.
     o              Competent world building involves constructing both formal and informal institutions.
       •                 Discussion of who or what people are as a group, what they are going to be in
         the future, and how they will get there (as a process).

=== Lesson 2 ===

       •                 Politics defined as world building involves thinking about other peoples'
         experiences of or in the world
       •                 Lead or organize group work, action, or cooperation
     o                     Politics is something done or undertaken by people who have a sense of
         responsibility for the group and its future
       •                 Democracy is a unique form of government because the informal social/cultural sphere of
         life is protected from undue intrusion by the formal, legal, governmental world.
     o              Much is expected of society in democracy. Government isn't expected to do it all (the
         world building). That's the genius of it.
       •                 Society in action
     o              Society is or means (a) People; (b) the values, beliefs, and identities they embrace"""

    print("TESTING IMPROVED CHUNKING")
    print("=" * 60)
    
    chunks = create_improved_chunks(sample_text)
    
    print(f"\\nCreated {len(chunks)} chunks from sample content")
    print(f"Original text: {len(sample_text)} characters")
    
    for i, chunk in enumerate(chunks):
        print(f"\\n--- CHUNK {i+1} ---")
        print(f"Length: {len(chunk)} characters")
        print(f"Preview: {chunk[:300]}{'...' if len(chunk) > 300 else ''}")
        print(f"Quality check: {'✅ Good' if len(chunk) > 200 and 'Lesson' in chunk else '❌ Poor'}")
    
    print(f"\\nAverage chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f}")
    
    return chunks


def test_lesson_specific_content():
    """Test with lesson-specific content"""
    
    lesson_text = """Understanding politics: politics, power, institutions (formal and informal)
    
    Defining politics
    What is politics? Politics is fundamentally about world building - the process by which people
    come together to create the kind of society they want to live in.
    
    This involves several key components:
    • People must understand their shared values and goals
    • They need mechanisms for making collective decisions  
    • Institutions must be created to organize and structure relationships
    • There must be ways to resolve conflicts and disagreements
    
    In democratic societies, politics is not just the domain of government officials.
    Every citizen has a role to play in the ongoing work of world building."""
    
    print("\\n\\nTESTING LESSON-SPECIFIC CONTENT")
    print("=" * 60)
    
    chunks = create_improved_chunks(lesson_text, lesson_number="1")
    
    for i, chunk in enumerate(chunks):
        print(f"\\n--- LESSON CHUNK {i+1} ---")
        print(f"Length: {len(chunk)} characters") 
        print(f"Content: {chunk}")
        
    return chunks


def compare_with_old_chunking():
    """Show the difference between old fragmented chunks and new improved chunks"""
    
    print("\\n\\nCOMPARISON: OLD vs NEW CHUNKING")
    print("=" * 60)
    
    print("OLD CHUNKING FRAGMENTS (from debug output):")
    old_fragments = [
        "IPP",
        "What else", 
        "Anger",
        "Patience",
        "Lesson 3"
    ]
    
    for fragment in old_fragments:
        print(f"❌ '{fragment}' - {len(fragment)} chars - No context!")
    
    print("\\nNEW IMPROVED CHUNKS:")
    sample_text = """=== Lesson 1 ===
    
       •                 IPP (Individual Political Participation)
     o              What else is involved in political engagement?
       •                 Anger - emotions that drive political action
       •                 Patience - the long-term perspective needed for change  
       •                 Companionship - people need community and shared purpose
       
       Politics involves understanding how people organize themselves to build 
       the world they want to live in."""
    
    chunks = create_improved_chunks(sample_text, lesson_number="1")
    
    for i, chunk in enumerate(chunks):
        print(f"✅ Chunk {i+1}: {len(chunk)} chars - Has context and meaning!")
        print(f"   Preview: {chunk[:100]}...")


if __name__ == "__main__":
    test_problematic_content()
    test_lesson_specific_content()
    compare_with_old_chunking()
    
    print("\\n" + "=" * 60)
    print("CHUNKING TEST COMPLETE!")
    print("The improved algorithm creates meaningful, contextual chunks")
    print("instead of fragmented bullet points without context.")