"""Check how well the REX Assistant's retriever matches FAQs and courses, and tune its thresholds.

    python scripts/eval_retriever.py                    # current retriever (REX_RETRIEVER, default rag)
    REX_RETRIEVER=tfidf python scripts/eval_retriever.py

Prints the top score, the runner-up and whether the answer is right for paraphrased questions
(English + Arabic) and for off-topic questions that must NOT match anything. Use the printed scores
to set REX_FAQ_MIN_SCORE / REX_FAQ_MARGIN / REX_COURSE_MIN_SCORE (or THRESHOLDS in retriever.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rex_ai.agents.rex.knowledge_base import get_kb  # noqa: E402
from rex_ai.agents.rex.retriever import make_retriever  # noqa: E402

FAQ_QUERIES = [
    ("will I receive a certificate at the end", "FAQ-01"),
    ("هل في شهادة بعد الكورس", "FAQ-01"),
    ("is this course accredited by the government", "FAQ-02"),
    ("can I split the fees into monthly payments", "FAQ-03"),
    ("ممكن ادفع بالتقسيط", "FAQ-03"),
    ("do you accept credit cards or vodafone cash", "FAQ-04"),
    ("can I get my money back if I cancel", "FAQ-05"),
    ("ايه سياسة استرجاع الفلوس", "FAQ-05"),
    ("what's the difference between hybrid and onsite", "FAQ-06"),
    ("will the lectures be recorded", "FAQ-07"),
    ("do I have to take a level test first", "FAQ-08"),
    ("are classes in arabic or english", "FAQ-09"),
    ("will you help me find a job", "FAQ-10"),
    ("هل بتساعدوا في الشغل بعد الكورس", "FAQ-10"),
    ("do I need to bring a laptop", "FAQ-11"),
    ("can you train our company team", "FAQ-12"),
    ("what happens to my phone number", "FAQ-13"),
]
OFF_TOPIC = ["what's the weather in alexandria", "who won the match yesterday", "tell me a joke",
             "how do I cook pasta", "ما هو عاصمة فرنسا"]
COURSE_QUERIES = [("I want to learn machine learning", "AI-101"), ("large language models and chatbots", "AI-201"),
                  ("power bi dashboards", "DS-101"), ("kubernetes", "CL-201"), ("اختبار البرمجيات", "SW-201")]


def main():
    r = make_retriever(get_kb().chunks)
    th = r.thresholds
    print(f"retriever={type(r).__name__} backend={r.backend} thresholds={th}\n")

    ok = 0
    print("FAQ paraphrases (should match):")
    for q, want in FAQ_QUERIES:
        hits = r.search(q, k=2, filters={"type": "faq"})
        top, second = hits[0].score, (hits[1].score if len(hits) > 1 else 0.0)
        got = hits[0].chunk.metadata["faq_id"]
        accepted = top >= th["faq_min"] and top - second >= th["faq_margin"]
        good = accepted and got == want
        ok += good
        print(f"  {'OK ' if good else 'XX '} top={top:.3f} gap={top - second:.3f} got={got} want={want}  {q}")

    fp = 0
    print("\nOff-topic (should NOT match):")
    for q in OFF_TOPIC:
        hits = r.search(q, k=2, filters={"type": "faq"})
        top, second = hits[0].score, (hits[1].score if len(hits) > 1 else 0.0)
        accepted = top >= th["faq_min"] and top - second >= th["faq_margin"]
        fp += accepted
        print(f"  {'XX ' if accepted else 'OK '} top={top:.3f} gap={top - second:.3f}  {q}")

    cok = 0
    print("\nCourse fallback:")
    for q, want in COURSE_QUERIES:
        hits = r.search(q, k=1, filters={"type": "course"})
        got, score = hits[0].chunk.metadata["course_id"], hits[0].score
        good = got == want and score >= th["course_min"]
        cok += good
        print(f"  {'OK ' if good else 'XX '} score={score:.3f} got={got} want={want}  {q}")

    print(f"\nFAQ: {ok}/{len(FAQ_QUERIES)} correct | off-topic false matches: {fp}/{len(OFF_TOPIC)} | "
          f"courses: {cok}/{len(COURSE_QUERIES)}")


if __name__ == "__main__":
    main()