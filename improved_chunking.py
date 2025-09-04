"""
Improved Chunking Algorithm

Creates meaningful, contextual chunks instead of fragmented bullet points.
Preserves hierarchical structure while ensuring semantic coherence.
"""

import re
from typing import List


class ImprovedChunker:
    """Improved chunking that preserves context and meaning"""
    
    def __init__(self, min_chunk_size=400, target_size=800, max_size=1200):
        self.min_chunk_size = min_chunk_size
        self.target_size = target_size
        self.max_size = max_size
    
    def create_contextual_chunks(self, text: str, lesson_number: str = None) -> List[str]:
        """Create chunks that preserve context and meaning"""
        
        if not text.strip():
            return []
        
        # Split into meaningful sections
        sections = self._split_into_sections(text)
        chunks = []
        
        for section in sections:
            if len(section) < self.min_chunk_size:
                # Small section - combine with context
                contextual_chunk = self._add_context_to_small_section(section, text, lesson_number)
                if contextual_chunk and len(contextual_chunk.strip()) >= 100:  # Only add meaningful chunks
                    chunks.append(contextual_chunk)
            elif len(section) <= self.max_size:
                # Good size - use as is but add lesson context
                contextual_chunk = self._add_lesson_context(section, lesson_number)
                chunks.append(contextual_chunk)
            else:
                # Too large - split while preserving meaning
                sub_chunks = self._split_large_section(section, lesson_number)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""
        
        # Look for natural breaks - lesson headers and major concepts
        lesson_pattern = r'(=== Lesson \d+(?:\.\d+)? ===)'
        
        # Split on lesson headers first
        if '=== Lesson' in text:
            parts = re.split(lesson_pattern, text)
            sections = []
            current_lesson_header = None
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                if part.startswith('=== Lesson'):
                    current_lesson_header = part
                else:
                    # This is lesson content
                    if current_lesson_header:
                        lesson_content = current_lesson_header + '\n\n' + part
                        sections.extend(self._split_lesson_content(lesson_content))
                    else:
                        sections.extend(self._split_lesson_content(part))
            
            if sections:
                return sections
        
        # If no lesson structure, split by major bullet points
        major_bullet_pattern = r'\n\s*•\s+[A-Z][^•]*?(?=\n\s*•|\n\s*$)'
        sections = re.split(major_bullet_pattern, text, flags=re.DOTALL)
        
        # Clean sections
        clean_sections = []
        for section in sections:
            section = section.strip()
            if section and len(section) > 50:
                clean_sections.append(section)
        
        return clean_sections if clean_sections else [text]
    
    def _split_lesson_content(self, lesson_content: str) -> List[str]:
        """Split content within a lesson while preserving structure"""
        
        # If the lesson content is reasonable size, keep it together
        if len(lesson_content) <= self.max_size:
            return [lesson_content]
        
        # For large lessons, try to split on major topic boundaries
        # Look for patterns that indicate new major topics
        split_patterns = [
            r'\n\s*•\s+[A-Z]',  # Major bullet points starting with capital letters
            r'\n\s*o\s+[A-Z]',  # Sub-bullets with capital letters (major sub-topics)
        ]
        
        best_sections = [lesson_content]
        
        for pattern in split_patterns:
            new_sections = []
            for section in best_sections:
                if len(section) > self.max_size:
                    # Find split points
                    split_points = [match.start() for match in re.finditer(pattern, section)]
                    
                    if split_points:
                        # Split at these points
                        last_end = 0
                        for split_point in split_points:
                            if split_point > last_end:
                                part = section[last_end:split_point].strip()
                                if part and len(part) > 100:
                                    new_sections.append(part)
                            last_end = split_point
                        
                        # Add final part
                        final_part = section[last_end:].strip()
                        if final_part and len(final_part) > 100:
                            new_sections.append(final_part)
                    else:
                        # No good split points, keep as is
                        new_sections.append(section)
                else:
                    new_sections.append(section)
            
            best_sections = new_sections
            
            # Check if we have reasonable sized sections now
            avg_size = sum(len(s) for s in best_sections) / len(best_sections)
            if avg_size <= self.target_size:
                break
        
        return best_sections
    
    def _add_context_to_small_section(self, section: str, full_text: str, lesson_number: str) -> str:
        """Add context to small sections to make them meaningful"""
        
        # Find surrounding context
        section_start = full_text.find(section)
        if section_start == -1:
            return self._add_lesson_context(section, lesson_number)
        
        # Get context before and after
        context_before = full_text[max(0, section_start - 400):section_start]
        context_after = full_text[section_start + len(section):section_start + len(section) + 400]
        
        # Build contextual chunk
        contextual_chunk = ""
        
        # Add lesson context
        if lesson_number:
            contextual_chunk += f"Lesson {lesson_number}:\n\n"
        
        # Add preceding context if meaningful
        if context_before.strip():
            context_before = context_before.strip()
            # Look for the last complete thought
            last_bullet = context_before.rfind('•')
            last_sub_bullet = context_before.rfind('o')
            last_break = max(last_bullet, last_sub_bullet)
            
            if last_break > 0 and len(context_before) - last_break < 300:
                contextual_chunk += "Context: " + context_before[last_break:] + "\n\n"
        
        # Add the main section
        contextual_chunk += section
        
        # Add following context if meaningful and space allows
        if context_after.strip() and len(contextual_chunk) < self.target_size:
            context_after = context_after.strip()
            remaining_space = self.target_size - len(contextual_chunk)
            
            if remaining_space > 100:
                # Find the next complete thought
                next_bullet = context_after.find('•')
                if next_bullet > 0 and next_bullet < remaining_space:
                    contextual_chunk += "\n\nContinues: " + context_after[:next_bullet]
                else:
                    contextual_chunk += "\n\nContinues: " + context_after[:remaining_space]
        
        return contextual_chunk
    
    def _add_lesson_context(self, section: str, lesson_number: str) -> str:
        """Add lesson context to a section"""
        if lesson_number and not section.startswith(f"Lesson {lesson_number}"):
            return f"Lesson {lesson_number}:\n\n{section}"
        return section
    
    def _split_large_section(self, section: str, lesson_number: str) -> List[str]:
        """Split large sections while preserving meaning"""
        
        chunks = []
        
        # Try to split on sub-bullets first
        sub_bullet_pattern = r'\n\s*o\s+'
        parts = re.split(sub_bullet_pattern, section)
        
        if len(parts) > 1:
            # Successfully split on sub-bullets
            current_chunk = ""
            
            for i, part in enumerate(parts):
                part = part.strip()
                if i > 0:  # Add back the bullet point marker
                    part = "    o " + part
                
                if len(current_chunk + part) <= self.max_size:
                    current_chunk += "\n" + part if current_chunk else part
                else:
                    if current_chunk:
                        chunks.append(self._add_lesson_context(current_chunk, lesson_number))
                    current_chunk = part
            
            if current_chunk:
                chunks.append(self._add_lesson_context(current_chunk, lesson_number))
        
        else:
            # Split on paragraphs as last resort
            paragraphs = section.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk + paragraph) <= self.max_size:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks.append(self._add_lesson_context(current_chunk, lesson_number))
                    current_chunk = paragraph
            
            if current_chunk:
                chunks.append(self._add_lesson_context(current_chunk, lesson_number))
        
        return chunks


# Convenience function for easy integration
def create_improved_chunks(text: str, lesson_number: str = None, 
                          min_size: int = 400, target_size: int = 800, max_size: int = 1200) -> List[str]:
    """Create improved contextual chunks from text"""
    chunker = ImprovedChunker(min_size, target_size, max_size)
    return chunker.create_contextual_chunks(text, lesson_number)