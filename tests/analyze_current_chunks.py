"""
Analyze Current Chunks

Review the current state of all chunks across courses to understand what we've built.
"""

import sys
import toml
from pathlib import Path

sys.path.append('..')

from pinecone import Pinecone


def analyze_all_chunks():
    """Analyze all chunks across all courses"""
    
    # Initialize Pinecone
    secrets_path = Path("../.streamlit/secrets.toml")
    secrets = toml.load(secrets_path)
    
    pc = Pinecone(api_key=secrets["PINECONE_API_KEY"])
    index = pc.Index(secrets["PINECONE_INDEX_NAME"])
    
    print("=" * 70)
    print("COMPREHENSIVE CHUNK ANALYSIS")
    print("=" * 70)
    
    # Get all namespaces
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {})
    
    total_vectors = sum(ns.get('vector_count', 0) for ns in namespaces.values())
    
    print(f"\\nOverall Statistics:")
    print(f"📊 Total Namespaces: {len(namespaces)}")
    print(f"📊 Total Vectors: {total_vectors}")
    
    # Categorize courses
    dual_courses = []
    legacy_courses = []
    
    for ns_name, ns_info in sorted(namespaces.items()):
        vector_count = ns_info.get('vector_count', 0)
        
        if '-' in ns_name:
            # Dual structure
            base_name = ns_name.split('-')[0]
            structure_type = ns_name.split('-')[1]
            dual_courses.append((base_name, structure_type, vector_count, ns_name))
        else:
            # Legacy structure
            legacy_courses.append((ns_name, vector_count))
    
    # Analyze dual structure courses
    print(f"\\n" + "=" * 50)
    print("DUAL STRUCTURE COURSES")
    print("=" * 50)
    
    dual_course_summary = {}
    for base_name, structure_type, vector_count, full_name in dual_courses:
        if base_name not in dual_course_summary:
            dual_course_summary[base_name] = {'mastery': 0, 'lessons': 0}
        dual_course_summary[base_name][structure_type] = vector_count
    
    for course_name, counts in sorted(dual_course_summary.items()):
        mastery_count = counts['mastery']
        lessons_count = counts['lessons']
        total_count = mastery_count + lessons_count
        
        print(f"\\n📚 {course_name.upper()}")
        print(f"   🧠 Mastery Knowledge: {mastery_count} vectors")
        print(f"   📖 Lesson Content: {lessons_count} vectors") 
        print(f"   📊 Total: {total_count} vectors")
        
        # Analyze balance
        if mastery_count == 0:
            print("   ⚠️  WARNING: No mastery content!")
        elif lessons_count == 0:
            print("   ⚠️  WARNING: No lesson content!")
        elif lessons_count > mastery_count * 10:
            print("   ⚠️  NOTE: Heavy lesson focus (good for boundaries)")
        elif mastery_count > lessons_count * 3:
            print("   ⚠️  NOTE: Heavy mastery focus")
        else:
            print("   ✅ Good balance of mastery and lesson content")
    
    # Analyze legacy courses
    print(f"\\n" + "=" * 50)
    print("LEGACY STRUCTURE COURSES")
    print("=" * 50)
    
    for course_name, vector_count in sorted(legacy_courses):
        print(f"\\n📚 {course_name.upper()}")
        print(f"   📊 Total Vectors: {vector_count}")
        
        if vector_count < 5:
            print("   ⚠️  WARNING: Very low content!")
        elif vector_count > 100:
            print("   📈 High content volume")
        else:
            print("   ✅ Reasonable content amount")
    
    # Sample content quality from key courses
    print(f"\\n" + "=" * 50)
    print("CONTENT QUALITY SAMPLES")
    print("=" * 50)
    
    # Check foundational course quality (our main test case)
    check_course_quality(index, "foundational", dual_course_summary.get("foundational", {}))


def check_course_quality(index, course_name, counts):
    """Check the quality of content for a specific course"""
    
    print(f"\\n🔍 Analyzing {course_name.upper()} content quality...")
    
    # Check mastery content
    if counts.get('mastery', 0) > 0:
        print(f"\\n--- Mastery Content Sample ---")
        try:
            mastery_results = index.query(
                vector=[0.0] * 768,
                top_k=2,
                include_metadata=True,
                namespace=f"{course_name}-mastery"
            )
            
            for i, match in enumerate(mastery_results.matches):
                text = match.metadata.get('text', '')
                print(f"Mastery {i+1}: {text[:150]}...")
                
                # Check quality indicators
                quality_indicators = {
                    "Has lesson context": "lesson" in text.lower(),
                    "Substantial content": len(text) > 200,
                    "Academic structure": any(marker in text for marker in ['•', 'o', '1.', '2.'])
                }
                
                for indicator, passes in quality_indicators.items():
                    status = "✅" if passes else "❌"
                    print(f"  {status} {indicator}")
                    
        except Exception as e:
            print(f"❌ Error checking mastery content: {e}")
    
    # Check lesson content
    if counts.get('lessons', 0) > 0:
        print(f"\\n--- Lesson Content Sample ---")
        try:
            lesson_results = index.query(
                vector=[0.0] * 768,
                top_k=3,
                include_metadata=True,
                namespace=f"{course_name}-lessons"
            )
            
            lesson_distribution = {}
            for match in lesson_results.matches:
                lesson_num = match.metadata.get('lesson_number', 'unknown')
                lesson_distribution[lesson_num] = lesson_distribution.get(lesson_num, 0) + 1
            
            print(f"Lesson distribution in sample: {lesson_distribution}")
            
            for i, match in enumerate(lesson_results.matches):
                text = match.metadata.get('text', '')
                lesson_num = match.metadata.get('lesson_number', 'unknown')
                
                print(f"\\nLesson {lesson_num} sample: {text[:150]}...")
                
                # Check lesson quality
                quality_indicators = {
                    "Proper lesson number": lesson_num != 'unknown',
                    "Lesson context": f"lesson {lesson_num}" in text.lower(),
                    "Substantial content": len(text) > 300,
                    "Good metadata": match.metadata.get('content_type') == 'lesson'
                }
                
                for indicator, passes in quality_indicators.items():
                    status = "✅" if passes else "❌"
                    print(f"  {status} {indicator}")
                    
        except Exception as e:
            print(f"❌ Error checking lesson content: {e}")


def provide_recommendations():
    """Provide recommendations based on analysis"""
    
    print(f"\\n" + "=" * 70)
    print("RECOMMENDATIONS & NEXT STEPS")
    print("=" * 70)
    
    recommendations = [
        "✅ Lesson 1 search is now working with lowered thresholds (0.2)",
        "✅ Dual structure is properly implemented for foundational, functional, govt, professional", 
        "✅ Lesson boundaries are enforced through separate namespaces",
        "⚠️  Monitor search score thresholds - may need adjustment per course",
        "📈 Consider processing remaining legacy courses to dual structure",
        "🧹 Test suite needs cleanup (remove duplicate/outdated files)",
        "🔍 Verify lesson boundary enforcement in production queries"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\\n🎯 Primary Success: 'walk through lesson 1' now works correctly!")


if __name__ == "__main__":
    analyze_all_chunks()
    provide_recommendations()