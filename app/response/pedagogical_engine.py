"""
Pedagogical Engine - Simplified

Handles format profiles for different question types.
"""

from typing import Dict


class PedagogicalEngine:
    """Manages format profiles for different question types"""

    def __init__(self):
        self.format_profiles = self._initialize_format_profiles()

    def _initialize_format_profiles(self) -> Dict[str, Dict[str, str]]:
        """Initialize format profiles for different question types"""
        return {
            "comparison": {
                "name": "comparison",
                "goal": "Clear, skimmable comparison",
                "directives": """\
- Start with a 1–2 sentence overview.
- Then a 2–6 row **Markdown table** for key aspects.
- Follow with **bulleted pros/cons** and a 1-line "Choose X if / Choose Y if".\
""",
                "exemplar": """\
### TL;DR
| Aspect | X | Y |
|---|---|---|
| Core idea | … | … |
| Best when | … | … |

- **X pros:** …
- **Y pros:** …

**Choose X if…** | **Choose Y if…**
"""
            },
            "definition": {
                "name": "definition",
                "goal": "Crisp definition, then essentials",
                "directives": """\
- Open with a bold **Definition** line (1 sentence).
- 2–3 short paragraphs for nuance.
- If variants exist in-course, list as bullets.\
""",
                "exemplar": """\
**Definition:** …

**Key characteristics**
- …
- …
"""
            },
            "process": {
                "name": "process",
                "goal": "Ordered steps",
                "directives": """\
- One-paragraph overview.
- **Numbered list** of steps, each 1–2 lines.\
""",
                "exemplar": """\
**Overview:** …

1. Step — …
2. Step — …
3. Step — …
"""
            },
            "narrative": {
                "name": "narrative",
                "goal": "Readable academic prose",
                "directives": "- Use short paragraphs; add a sub-header only if the content naturally splits.",
                "exemplar": ""
            }
        }

    def select_format_profile(self, question: str) -> Dict[str, str]:
        """Select appropriate format profile based on question"""
        q = question.lower()

        # Comparison questions
        if any(k in q for k in ["compare", " vs ", "difference between"]):
            return self.format_profiles["comparison"]

        # Definition questions
        if any(k in q for k in ["what is", "define"]):
            return self.format_profiles["definition"]

        # Process/how-to questions
        if any(k in q for k in ["how does", "process", "steps"]):
            return self.format_profiles["process"]

        # Default narrative
        return self.format_profiles["narrative"]
