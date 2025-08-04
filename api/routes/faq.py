from fastapi import APIRouter, HTTPException
from services.llm.base import LLMService
from config import settings
from pymongo import MongoClient
from collections import defaultdict
import re
import logging
import json
from typing import List, Dict

router = APIRouter(tags=["FAQ"])
logger = logging.getLogger(__name__)
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
conversations_collection = db["conversations"]


class FAQGenerator:
    def __init__(self):
        self.llm = LLMService()
        self.min_occurrences = 2
        self.max_faqs = 6
        self.stop_words = {
            "what",
            "how",
            "can",
            "you",
            "me",
            "the",
            "a",
            "an",
            "is",
            "are",
            "do",
            "does",
            "could",
            "would",
            "please",
        }

    def normalize_question(self, text: str) -> str:
        """Improved text normalization"""
        if not text:
            return ""
        text = re.sub(r"[^\w\s]", "", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split() if w not in self.stop_words]
        return " ".join(words)

    def group_similar_questions(self, questions: List[str]) -> Dict[str, List[str]]:
        """Group similar questions using LLM with better error handling"""
        if not questions or len(questions) < 2:
            return {q: [q] for q in questions}

        prompt = f"""
        Analyze these wealth management questions and group similar ones together:
        {questions[:50]}
        
        Return ONLY a valid JSON dictionary where:
        - Key is the most representative question from each group
        - Value is list of all similar questions in that group
        
        Example output format:
        {{
            "What is my portfolio performance?": [
                "how is my portfolio doing",
                "show me my portfolio performance"
            ],
            "What are my investment options?": [
                "what can I invest in",
                "show investment choices"
            ]
        }}
        """

        try:
            response = self.llm.generate_response(prompt)
            if not response:
                raise ValueError("Empty LLM response")

            response = response.strip()
            if not response.startswith("{"):
                response = "{" + response.split("{", 1)[-1]
            if not response.endswith("}"):
                response = response.split("}", 1)[0] + "}"

            return json.loads(response)
        except Exception as e:
            logger.warning(f"LLM grouping failed, using fallback method: {str(e)}")
            groups = defaultdict(list)
            for q in questions:
                key = " ".join(q.split()[:3]).lower()
                groups[key].append(q)
            return groups

    def get_representative_question(self, group: List[str]) -> str:
        """Select the best phrased question from a group"""
        if not group:
            return ""
        for q in sorted(group, key=lambda x: -len(x), reverse=True):
            if len(q.split()) >= 4 and q.strip().endswith("?"):
                return q
        return group[0]


@router.get("/faq", summary="Get frequently asked questions")
async def get_faqs():
    """Get intelligent FAQs grouped by semantic similarity"""
    try:
        generator = FAQGenerator()
        question_counter = defaultdict(int)
        raw_questions = []

        cursor = conversations_collection.find(
            {"messages.role": "user"}, {"messages": {"$elemMatch": {"role": "user"}}}
        ).limit(1000)

        for convo in cursor:
            for msg in convo.get("messages", []):
                question = msg.get("content", "").strip()
                if question and len(question.split()) >= 3:
                    raw_questions.append(question)
                    norm_question = generator.normalize_question(question)
                    question_counter[norm_question] += 1

        frequent_questions = [
            q
            for q in question_counter
            if question_counter[q] >= generator.min_occurrences
        ]

        if not frequent_questions:
            return {
                "faqs": [
                    "List down our global customers",
                    "Explain our frontend tech stack",
                    "Explain digital experience management in short",
                    "Can you explain user experience practice at Tech Mahindra positively affects?",
                ],
                "message": "Insufficient question data",
                "stats": {
                    "total_questions_analyzed": len(raw_questions),
                    "unique_questions": len(question_counter),
                    "frequent_question_groups": 0,
                },
            }

        question_groups = generator.group_similar_questions(frequent_questions)

        faqs = []
        for representative, group in question_groups.items():
            total_occurrences = sum(question_counter.get(q, 0) for q in group)
            if total_occurrences >= generator.min_occurrences:
                candidate_questions = [
                    q for q in raw_questions if generator.normalize_question(q) in group
                ]
                best_question = generator.get_representative_question(
                    candidate_questions
                )
                if best_question:
                    faqs.append(
                        {
                            "question": best_question,
                            "occurrences": total_occurrences,
                            "variations": len(group),
                        }
                    )

        faqs.sort(key=lambda x: (-x["occurrences"], -x["variations"]))
        top_faqs = [f["question"] for f in faqs[: generator.max_faqs]]

        default_faqs = [
            "List down our global customers",
            "Explain our frontend tech stack",
            "Explain digital experience management in short",
            "Can you explain user experience practice at Tech Mahindra positively affects?",
        ]

        if len(top_faqs) < 4:
            needed = 4 - len(top_faqs)
            top_faqs.extend(default_faqs[:needed])

        return {
            "faqs": top_faqs,
            "stats": {
                "total_questions_analyzed": len(raw_questions),
                "unique_questions": len(question_counter),
                "frequent_question_groups": len(faqs),
            },
        }

    except Exception as e:
        logger.error(f"FAQ generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Could not generate FAQs. Please try again later."
        )
