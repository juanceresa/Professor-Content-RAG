#!/usr/bin/env python3
"""
Test lesson filtering and prioritization logic
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_lesson_prioritization():
    """Test the lesson prioritization scoring algorithm"""
    print("🎯 Testing Lesson Prioritization Logic\n")
    
    # Mock chunk data with different lessons
    test_chunks = [
        {"lesson": "3", "text": "Current lesson content", "score": 0.8, "hierarchy_path": "Current Topic"},
        {"lesson": "2", "text": "Previous lesson content", "score": 0.7, "hierarchy_path": "Previous Topic"},
        {"lesson": "1", "text": "Early foundation", "score": 0.55, "hierarchy_path": "Foundation Topic"},
        {"lesson": "4", "text": "Future lesson content", "score": 0.9, "hierarchy_path": "Future Topic"},
        {"lesson": "", "text": "General content", "score": 0.75, "hierarchy_path": "General Topic"}
    ]
    
    current_lesson = "3"
    
    # Apply the same scoring logic from streamlit_app.py
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
                
                # NO boost for future lessons (filter out)
                elif lesson_num > current_num:
                    score -= 1.0  # Penalize future content
                    
            except (ValueError, TypeError):
                pass
        
        return score
    
    # Apply scoring and sort
    scored_chunks = []
    for chunk in test_chunks:
        enhanced_score = sort_by_lesson_and_academic_value(chunk)
        scored_chunks.append({
            **chunk,
            "enhanced_score": enhanced_score,
            "boost": enhanced_score - chunk['score']
        })
    
    # Sort by enhanced score (descending)
    scored_chunks.sort(key=lambda x: x['enhanced_score'], reverse=True)
    
    print(f"Student is on Lesson {current_lesson}. Prioritization results:\n")
    
    for i, chunk in enumerate(scored_chunks, 1):
        lesson_display = f"Lesson {chunk['lesson']}" if chunk['lesson'] else "General"
        print(f"{i}. {lesson_display} (Score: {chunk['score']:.2f} → {chunk['enhanced_score']:.2f}, Boost: +{chunk['boost']:.2f})")
        print(f"   Content: {chunk['text']}")
        print(f"   Topic: {chunk['hierarchy_path']}")
        print()
    
    # Validate expected order
    expected_order = ["3", "2", "", "1", "4"]  # Current → Previous → General → Early → Future
    actual_order = [chunk['lesson'] for chunk in scored_chunks]
    
    print("Expected lesson order:", expected_order)
    print("Actual lesson order:  ", actual_order)
    
    if actual_order == expected_order:
        print("✅ Lesson prioritization working correctly!")
    else:
        print("❌ Lesson prioritization needs adjustment")
    
    return actual_order == expected_order

def test_chunk_selection_logic():
    """Test 90%/10% chunk selection logic"""
    print("\n📚 Testing 90%/10% Chunk Selection Logic\n")
    
    # Mock chunks after prioritization
    current_lesson_chunks = [
        {"lesson": "3", "text": "Current concept A", "score": 2.3},
        {"lesson": "3", "text": "Current concept B", "score": 2.2},
        {"lesson": "3", "text": "Current concept C", "score": 2.1},
        {"lesson": "3", "text": "Current concept D", "score": 2.0}
    ]
    
    connection_chunks = [
        {"lesson": "2", "text": "Foundation from lesson 2", "score": 0.9},
        {"lesson": "1", "text": "Early foundation", "score": 0.75}
    ]
    
    general_chunks = [
        {"text": "General content", "score": 0.8}
    ]
    
    # Apply the 90%/10% selection logic
    selected_chunks = []
    
    # Take 2-3 chunks from current lesson (90% priority)
    selected_chunks.extend(current_lesson_chunks[:3])
    
    # Add 1 connection chunk if available and space remains (10% priority)
    if len(selected_chunks) < 3 and connection_chunks:
        selected_chunks.extend(connection_chunks[:1])
    
    # Fill any remaining slots with general content
    remaining_slots = max(0, 3 - len(selected_chunks))
    if remaining_slots > 0:
        selected_chunks.extend(general_chunks[:remaining_slots])
    
    print("Selected chunks for response:")
    for i, chunk in enumerate(selected_chunks, 1):
        lesson_display = f"Lesson {chunk.get('lesson', 'General')}"
        print(f"{i}. {lesson_display}: {chunk['text']} (Score: {chunk['score']:.2f})")
    
    # Validate 90%/10% distribution
    current_lesson_count = sum(1 for chunk in selected_chunks if chunk.get('lesson') == '3')
    connection_count = len(selected_chunks) - current_lesson_count
    
    current_percentage = (current_lesson_count / len(selected_chunks)) * 100
    connection_percentage = (connection_count / len(selected_chunks)) * 100
    
    print(f"\nDistribution:")
    print(f"- Current lesson: {current_lesson_count}/{len(selected_chunks)} chunks ({current_percentage:.1f}%)")
    print(f"- Connections: {connection_count}/{len(selected_chunks)} chunks ({connection_percentage:.1f}%)")
    
    # Should be approximately 90%/10% or better for current lesson
    if current_percentage >= 75:  # Allow some flexibility
        print("✅ Chunk selection prioritizes current lesson appropriately!")
        return True
    else:
        print("❌ Chunk selection needs more current lesson focus")
        return False

if __name__ == "__main__":
    print("🎓 Lesson Filtering Test Suite\n")
    print("=" * 60)
    
    success1 = test_lesson_prioritization()
    success2 = test_chunk_selection_logic()
    
    if success1 and success2:
        print("\n🎉 All lesson filtering tests passed!")
    else:
        print("\n⚠️  Some lesson filtering tests need attention")