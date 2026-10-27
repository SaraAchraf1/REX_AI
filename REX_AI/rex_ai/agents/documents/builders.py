"""Rule-based content for every template section (brief, Agent 4 items 26-35), English and Arabic.

    sections = Builder(facts, template, "en").build()

Each template section key maps to a method: s_<doc_type>__<key> if the type needs its own version, else s_<key>.
Everything is derived from CourseFacts (curriculum, catalogue, reference syllabi) - fees, dates, hours, weights and
instructors are copied, never estimated; what is missing is said to be missing. Rule-made question-bank items test
course structure and outcomes; with the LLM enabled they are rewritten as content questions (agent.py).
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Union

from rex_ai.agents.curriculum.alignment import ASSESSES, _lead_words, outcome_domain
from rex_ai.agents.curriculum.generator import ARTIFACT, ARTIFACT_AR, DEPARTMENT_AR, HOURS_PER_WEEK, INDUSTRY_AR, \
    LEVEL_AR, round_half
from rex_ai.agents.curriculum.pdf import COUNTRIES
from rex_ai.agents.curriculum.research import _words
from rex_ai.agents.documents import templates as tpl
from rex_ai.agents.documents.facts import CourseFacts
from rex_ai.agents.rex.text_utils import normalize
from rex_ai.core.schemas import DocSection, DocTable

log = logging.getLogger("rex.documents.builders")

SESSION_HOURS = 3
TIMING = (("open", 0.10), ("teach", 0.35), ("practice", 0.40), ("review", 0.15))      # facilitator guide pattern
LETTERS = {"en": "ABCD", "ar": "أبجد"}
TECH = {"AI", "Data", "Software", "Cloud", "Cybersecurity"}

BLOOM_VERBS = {
    "remember": ["define", "list", "recall", "identify", "state", "recognise", "recognize", "name", "يعرف", "يذكر",
                 "يحدد", "يسرد"],
    "understand": ["explain", "describe", "summarise", "summarize", "outline", "interpret", "discuss", "classify",
                   "يشرح", "يصف", "يفسر", "يلخص", "يوضح"],
    "apply": ["apply", "use", "implement", "perform", "build", "configure", "run", "prepare", "operate", "write",
              "test", "conduct", "deploy", "clean", "automate", "train", "calculate", "يطبق", "يستخدم", "ينفذ",
              "يبني", "يعد", "يجري", "يحسب"],
    "analyse": ["analyse", "analyze", "compare", "diagnose", "distinguish", "investigate", "examine", "triage",
                "يحلل", "يقارن", "يميز", "يشخص"],
    "evaluate": ["evaluate", "assess", "justify", "critique", "select", "choose", "recommend", "prioritise",
                 "prioritize", "يقيم", "يختار", "يوصي", "يبرر"],
    "create": ["create", "design", "develop", "plan", "produce", "construct", "compose", "present", "lead",
               "communicate", "يصمم", "يطور", "ينشئ", "يخطط", "يعرض", "يقود"],
}

P = {
    "en": {
        "none": "None", "tbc": "To be confirmed", "total": "Total", "hours": "hours", "h": "h", "weeks": "weeks",
        "module": "Module", "topics": "Topics", "practical": "Practical work", "assessment": "Assessment",
        "type": "Type", "weight": "Weight", "field": "Item", "value": "Details", "framework": "Framework",
        "coverage": "Coverage", "gaps": "Gaps to close", "lo": "LO",
        "bloom": {"remember": "Remember", "understand": "Understand", "apply": "Apply", "analyse": "Analyse",
                  "evaluate": "Evaluate", "create": "Create"},
        "formats": {"online_live": "Live online", "onsite": "On-site", "hybrid": "Hybrid"},
        "desc_compose": "{title} is a {hours:g}-hour {level} course in {dept} for {aud}.",
        "desc_aim": "It will help you {aim}.",
        "level_line": "Skill level: {level}.",
        "outcomes_intro": "By the end of the course you will be able to:",
        "outcomes_intro_formal": "On successful completion, participants will be able to:",
        "duration": "{hours:g} hours over about {weeks} weeks (around {per_week} hours a week)",
        "formats_line": "Training formats: {formats}",
        "language_line": "Language of instruction: {langs}",
        "next_cohorts": "Upcoming cohorts:",
        "cohort": "{date} - {fmt}, {days} {time}{loc} ({seats} seats left)",
        "no_cohort": "No cohort is scheduled yet - contact the academy for the next start date.",
        "instructor": "{name}, {title} - {years} years of experience. {bio}",
        "no_instructor": "Instructor to be announced.",
        "fee": "Course fee: {fee:,.0f} {cur}",
        "fee_promo": "Current offer: {name} - {disc:g}% off ({cond}), valid until {until}. Fee with the offer: "
                     "{price:,.0f} {cur}",
        "installments": "Payment in instalments is available.",
        "no_fee": "Contact the academy for fees and current offers.",
        "certificate_align": "The course content is aligned with {fw}. Alignment is not accreditation: no "
                             "accrediting body has reviewed this course.",
        "no_careers": "Career opportunities are not listed in the catalogue for this course.",
        # proposal
        "prepared_by": "Prepared by: Andalusia Academy - REX AI document service", "date_line": "Date: {date}",
        "client_default": "the client organisation",
        "exec_need": "{client} wants its {aud} to build practical skills in {topic}.",
        "exec_offer": "We propose {title}, a {hours:g}-hour {level} programme of {n} modules delivered over about "
                      "{weeks} weeks.",
        "exec_capstone": "Every participant completes a capstone project, so the programme ends with work the "
                         "organisation can use.",
        "exec_measure": "Progress is measured through {assess}.",
        "participants_level": "Level: {level}.",
        "participants_prereq": "Prerequisites: {pre}.",
        "method_practice": "About {pct}% of the training time is hands-on practice: labs, exercises and the "
                           "capstone project.",
        "method_formats": "Delivery: {formats}.",
        "method_bullets": ["Short instructor-led explanations followed by guided practice",
                           "Practical exercises on realistic {ind} cases", "Module quizzes to check understanding",
                           "Instructor feedback on every practical submission"],
        "schedule_line": "Duration: {hours:g} hours over about {weeks} weeks ({per_week} hours a week).",
        "schedule_client": "Exact dates are agreed with the client; the academy's upcoming public cohorts are:",
        "invest_fee": "Fee per participant (public course): {fee:,.0f} {cur}",
        "invest_group": "Group, in-house and tender pricing is confirmed in the commercial offer.",
        "quality_review": "The curriculum was reviewed and approved internally by {by} on {date}.",
        "quality_cov": "{fw}: {pct:g}% coverage",
        "quality_none": "No framework alignment report is available yet.",
        "disclaimer": "Alignment is not accreditation: these figures compare the course content with published "
                      "framework topics and do not mean the course is accredited by any body.",
        "next_steps": ["Confirm the number of participants and the preferred dates",
                       "Agree the capstone cases or datasets with the client's subject-matter owner",
                       "Confirm the commercial offer and sign the training agreement",
                       "Participants receive the joining instructions and the participant guide"],
        # curriculum
        "f_department": "Department", "f_audience": "Target audience", "f_level": "Skill level",
        "f_industry": "Industry", "f_duration": "Course duration", "f_country": "Country",
        "f_framework": "Accreditation framework (alignment only)", "f_prereq": "Prerequisites",
        "f_sources": "Data sources & update date", "f_modules": "Modules", "f_format": "Format",
        "f_assessments": "Assessments", "f_version": "Curriculum version",
        # participant guide
        "welcome": "Welcome to {title}. This guide goes with you through the course: it shows what each module "
                   "covers, the activities you will do and the questions to reflect on afterwards.",
        "welcome_use": "Bring it to every session, write your notes in it, and use the reflection questions to "
                       "prepare for the quizzes and the capstone project.",
        "module_intro": "This module takes about {h:g} hours. In it you will learn:",
        "module_no_topics": "Your instructor will cover {m} in detail; the key points from reference syllabi are "
                            "listed below.",
        "activities": "Activities", "reflect": "Reflection questions", "further": "Further reading",
        "key_points": "Key points",
        "activity_from_topic": "Practise {t}: follow your instructor's demonstration, then repeat it on your own "
                               "example and note what you changed",
        "reflect_q": ["Which idea from \"{m}\" will you use first in your own work, and how?",
                      "What was the most difficult part of this module, and what will you do to master it?",
                      "How would you explain {t} to a colleague in two sentences?"],
        "ref_point": "{topic}", "ref_outcome": "Reference outcome: {o}", "ref_cite": "Source: {c}",
        "pass_rule": "To pass you need at least {pass_mark}% overall and {attendance}% attendance, and you must "
                     "submit the capstone project.",
        "capstone_deliver": "You will hand in a project plan, the working {artifact}, a short report and a final "
                            "presentation.",
        "support_instructor": "Your instructor: {name} ({title})",
        "support_general": ["Ask questions during sessions or in the course discussion space",
                            "Book office hours with your instructor for help with the capstone",
                            "Contact the academy's training coordinator for scheduling or certificate questions"],
        "glance_modules": "{n} modules + capstone project",
        "default_plan_note": "Default academy assessment plan: the final weights are confirmed in the approved "
                             "curriculum.",
        # facilitator guide
        "purpose": "This guide helps you deliver {title} consistently. It gives the session plan, what to prepare, a "
                   "timed plan for every module, activity instructions, questions to check understanding and the "
                   "difficulties learners usually meet.",
        "purpose_2": "Learners use the participant guide; the question bank and rubrics are for you only.",
        "fac_profile": "A practitioner in {dept} with experience of {ind} work and of teaching adults at {level} "
                       "level.",
        "fac_assigned": "Instructors assigned in the catalogue:",
        "prep_common": ["Participant guides (printed or shared) and the attendance sheet",
                        "Question bank and assessment rubrics (keep confidential)",
                        "Presentation slides built from the presentation outline",
                        "Learning platform space for submissions and announcements"],
        "prep_tech": ["A lab machine or virtual lab per learner with the course tools installed and tested",
                      "Sample datasets / projects for every lab, with model solutions",
                      "Backup offline copies of the lab material in case of network problems"],
        "prep_case": ["Printed case studies and templates for every exercise", "Flip charts or a digital whiteboard "
                      "for group work"],
        "sessions": "Sessions ({n} h)", "main_activities": "Main activities", "segment": "Segment",
        "minutes": "Minutes", "what_to_do": "What to do",
        "seg": {"open": "Opening and recap", "teach": "Explain and demonstrate", "practice": "Guided practice",
                "review": "Review and check understanding"},
        "seg_do": {"open": "Recap the previous module, state today's objectives and why they matter in {ind} work",
                   "teach": "Explain {topics}; demonstrate each with a worked example",
                   "practice": "Learners do the activities in pairs; circulate, give feedback, note common errors",
                   "review": "Ask the check questions, run the module quiz, answer questions, preview the next module"},
        "mod_objectives": "By the end of this module learners can work with: {topics}.",
        "fac_activity": "{x} - brief the task (5 min), let learners work, then discuss two solutions with the group",
        "check_q": "Ask: \"Explain {t} in your own words and give an example from {ind}.\"",
        "difficulties": "Common difficulties", "check": "Check for understanding",
        "diff_generic": ["Learners copy the demonstration without understanding why: ask them to change one input "
                         "and predict the result", "Mixed levels in the group: pair stronger and newer learners",
                         "Time overruns in practice: keep the review segment, move extra exercises to homework"],
        "assess_admin": "{name} ({w:g}%): {how}",
        "assess_how": {"quiz": "run at the end of each module; 10-15 questions from the question bank; give the "
                               "results in the next session",
                       "exam": "closed-book, timed, questions drawn from the question bank across all modules",
                       "assignment": "set at the end of the module, due before the next one; mark with the rubric "
                                     "and return feedback within a week",
                       "project": "brief learners early; review the plan before building starts; mark with the "
                                  "capstone rubric",
                       "presentation": "10 minutes per learner or team plus questions; mark with the rubric"},
        "capstone_sup": ["Introduce the capstone brief in the first week so learners can collect ideas",
                         "Approve every project plan before building starts",
                         "Hold a short check-in at each milestone; record progress",
                         "Mark the final submission and presentation with the capstone rubric",
                         "Check for plagiarism and undeclared use of AI tools"],
        # slides
        "slide_title": "Course introduction", "slide_agenda": "Agenda", "slide_outcomes": "Learning outcomes",
        "slide_activity": "Activity: {m}", "slide_capstone": "Capstone project", "slide_summary": "Summary and Q&A",
        "slide_hours": "{hours:g} hours · {n} modules · {level}",
        "notes_title": "Welcome the group, introduce yourself and ask each learner what they want from the course.",
        "notes_agenda": "Walk through the modules and show where the capstone fits.",
        "notes_outcomes": "Read the outcomes and ask which one matters most for their work.",
        "notes_module": "Introduce the module: why it matters in {ind} work and what learners will be able to do.",
        "notes_topic": "Explain {t}; show one worked example from {ind}; ask a learner to explain it back.",
        "topic_bullets": ["What it is and why it matters", "How it is used in {ind} work", "Worked example",
                          "Common mistakes to avoid"],
        "notes_activity": "Brief the activity, set a time limit, then discuss two solutions.",
        "notes_capstone": "Present the brief, milestones and rubric; answer questions about scope.",
        "notes_summary": "Recap the outcomes, collect feedback and explain next steps (certificate, further study).",
        "summary_bullets": ["What we covered", "How you will be assessed", "Next steps after the course"],
        # question bank
        "usage": "Confidential - for instructors only. Questions are grouped by module; each one shows its type, "
                 "cognitive level and the learning outcome it checks. Draw module quizzes from the multiple-choice "
                 "and true/false items, and use the short-answer and practical items in assignments and exams.",
        "usage_rules": "Rule-generated items check knowledge of the course structure and outcomes; an instructor "
                       "must review them (and add content questions) before use.",
        "usage_llm": "Items were written by the LLM from the curriculum and reference syllabi; an instructor must "
                     "review every item before use.",
        "qt": {"mcq": "Multiple choice", "tf": "True / False", "short": "Short answer", "practical": "Practical"},
        "q_cols": ["#", "Type", "Level", "LO", "Question", "Options", "Answer / marking points"],
        "q_mcq": "Which of the following is a topic of the module \"{m}\"?",
        "q_mcq2": "The module \"{m}\" also covers which of these topics?",
        "q_tf": "\"{t}\" is part of the module \"{m}\".",
        "q_short": "Explain {t} and give an example from {ind} work.",
        "q_short_ans": "Correct explanation of {t} (2); relevant example (1)",
        "q_practical_ans": "Follows the steps (2); correct result (2); explains the choices made (1)",
        "q_ref_ans": "Marking points from the reference syllabus ({c})",
        "true": "True", "false": "False", "no_items": "Too few topics in this module to generate items - add "
                                                       "topics to the curriculum or write items by hand.",
        "bp_cols": ["Module", "Multiple choice", "True / False", "Short answer", "Practical", "Total"],
        # rubrics
        "levels": [("Excellent", 4, "Fully meets the criterion, with no errors and evidence of insight"),
                   ("Good", 3, "Meets the criterion with minor errors"),
                   ("Satisfactory", 2, "Partly meets the criterion; several errors or gaps"),
                   ("Needs improvement", 1, "Does not meet the criterion yet")],
        "scale_cols": ["Level", "Points", "Meaning"],
        "rubric_intro": "Weight: {w:g}% of the final grade ({pts:g} points). Outcomes assessed: {los}.",
        "rubric_cols": ["Criterion (points)", "Excellent (4)", "Good (3)", "Satisfactory (2)", "Needs improvement (1)"],
        "crit": {
            "project": [("Requirements and scope", "all requirements met and scope well defined",
                         "most requirements met", "several requirements missing", "requirements largely unmet"),
                        ("Correct application of course concepts", "concepts applied correctly and justified",
                         "minor errors in applying concepts", "several errors; limited justification",
                         "concepts misapplied"),
                        ("Quality of the deliverable", "complete, working and well organised",
                         "works with minor issues", "partly working or incomplete", "not working"),
                        ("Documentation and presentation", "clear, concise, answers questions confidently",
                         "clear with minor gaps", "hard to follow in places", "unclear or missing")],
            "assignment": [("Correctness", "all results correct", "minor errors", "several errors",
                            "mostly incorrect"),
                           ("Use of techniques and tools", "appropriate techniques used efficiently",
                            "appropriate with minor issues", "partly appropriate", "inappropriate"),
                           ("Analysis and interpretation", "insightful and well supported", "sound",
                            "superficial", "missing"),
                           ("Clarity of the submission", "well structured and easy to follow", "mostly clear",
                            "hard to follow", "disorganised")],
            "presentation": [("Content", "accurate and complete", "minor gaps", "several gaps", "inaccurate"),
                             ("Structure and timing", "logical, on time", "mostly logical", "hard to follow",
                              "no structure"),
                             ("Answers to questions", "confident and correct", "mostly correct", "partial",
                              "unable to answer")],
            "score": [("Score", "85-100%", "70-84%", "{pm}-69%", "below {pm}%")],
        },
        "grading": ["Pass mark: {pm}% of the total weighted score", "Minimum attendance: {att}%",
                    "The capstone project must be submitted to pass", "Assessment weights: {weights}"],
        # capstone brief
        "scenario": "You work for a {ind} organisation in {country}. {need} Your task is to plan, build and present "
                    "a solution that applies what you learned in this course.",
        "req_module": "Apply {m}{detail}",
        "req_general": ["Document your assumptions, data sources and design decisions",
                        "Present the result to a non-technical audience in 10 minutes"],
        "deliverables": ["Project plan (scope, approach, timeline) - approved by the instructor",
                         "The working {artifact}", "Short report (3-5 pages): problem, approach, results, "
                         "recommendations", "Final presentation and live demonstration"],
        "ms_cols": ["Milestone", "Course week", "Hours", "Output"],
        "ms": [("Plan", "Project plan approved"), ("Build", "Working {artifact}"),
               ("Review", "Peer and instructor feedback applied"), ("Present", "Report and presentation")],
        "submission": ["Submit all files on the learning platform before the final session",
                       "Name files: <course>_<your name>_<deliverable>",
                       "Present in the final session: 10 minutes plus 5 minutes of questions",
                       "Late submissions need the instructor's approval"],
        "integrity": ["The work you submit must be your own (or your team's, for team projects)",
                      "Cite every source, dataset and code you reuse",
                      "AI tools may help you learn and draft, but you must declare how you used them and be able to "
                      "explain every part of your work",
                      "Plagiarism or undeclared AI-generated work fails the capstone"],
        "capstone_default": "A capstone project in which learners apply the modules of {title} to a realistic "
                            "{ind} case.",
        # outcomes plan
        "plan_intro": "This plan shows how each learning outcome of {title} is taught and assessed, so that every "
                      "outcome is measured at least once and the weights add up to 100%.",
        "map_cols": ["LO", "Learning outcome", "Level", "Taught in", "Assessed by"],
        "as_cols": ["Assessment", "Type", "Weight", "Outcomes assessed"],
        "cov_ok": "Every outcome is assessed at least once.",
        "cov_missing": "Not assessed yet: LO{n} \"{o}\" - add it to an assessment.",
        "cov_verb": "LO{n} has no measurable action verb - rewrite it (e.g. \"Explain ...\", \"Build ...\").",
        "cov_weights": "Assessment weights add up to {w:g}% (should be 100%).",
        "cov_weights_ok": "Assessment weights add up to 100%.",
        "cov_all_modules": "all modules",
    },
    "ar": {
        "none": "لا يوجد", "tbc": "يُحدد لاحقاً", "total": "الإجمالي", "hours": "ساعة", "h": "س", "weeks": "أسابيع",
        "module": "الوحدة", "topics": "الموضوعات", "practical": "التطبيق العملي", "assessment": "التقييم",
        "type": "النوع", "weight": "الوزن", "field": "البند", "value": "التفاصيل", "framework": "الإطار",
        "coverage": "نسبة التغطية", "gaps": "فجوات يجب سدها", "lo": "ناتج",
        "bloom": {"remember": "تذكر", "understand": "فهم", "apply": "تطبيق", "analyse": "تحليل", "evaluate": "تقييم",
                  "create": "ابتكار"},
        "formats": {"online_live": "مباشر عبر الإنترنت", "onsite": "حضوري", "hybrid": "مدمج"},
        "desc_compose": "{title} دورة مدتها {hours:g} ساعة بمستوى {level} في مجال {dept}، موجهة إلى {aud}.",
        "desc_aim": "تساعدك الدورة على {aim}.",
        "level_line": "المستوى: {level}.",
        "outcomes_intro": "في نهاية الدورة ستكون قادراً على أن:",
        "outcomes_intro_formal": "عند إتمام الدورة بنجاح يكون المشارك قادراً على أن:",
        "duration": "{hours:g} ساعة على مدى {weeks} أسابيع تقريباً (نحو {per_week} ساعات أسبوعياً)",
        "formats_line": "أساليب التدريب: {formats}",
        "language_line": "لغة التدريس: {langs}",
        "next_cohorts": "الدفعات القادمة:",
        "cohort": "{date} - {fmt}، {days} {time}{loc} (المقاعد المتبقية: {seats})",
        "no_cohort": "لا توجد دفعة مجدولة حالياً - تواصل مع الأكاديمية لمعرفة موعد البدء القادم.",
        "instructor": "{name}، {title} - خبرة {years} سنة. {bio}",
        "no_instructor": "سيتم الإعلان عن المدرب لاحقاً.",
        "fee": "رسوم الدورة: {fee:,.0f} {cur}",
        "fee_promo": "العرض الحالي: {name} - خصم {disc:g}% ({cond})، ساري حتى {until}. الرسوم بعد الخصم: "
                     "{price:,.0f} {cur}",
        "installments": "يتوفر الدفع بالتقسيط.",
        "no_fee": "تواصل مع الأكاديمية لمعرفة الرسوم والعروض الحالية.",
        "certificate_align": "محتوى الدورة متوافق مع {fw}. المواءمة ليست اعتماداً رسمياً: لم تراجع أي جهة اعتماد هذه الدورة.",
        "no_careers": "لا توجد فرص وظيفية مدرجة في الكتالوج لهذه الدورة.",
        "prepared_by": "إعداد: أكاديمية الأندلس - خدمة المستندات في REX AI", "date_line": "التاريخ: {date}",
        "client_default": "الجهة العميلة",
        "exec_need": "تسعى {client} إلى تنمية المهارات العملية لدى {aud} في مجال {topic}.",
        "exec_offer": "نقترح برنامج {title}، وهو برنامج بمستوى {level} مدته {hours:g} ساعة يتكون من {n} وحدات "
                      "ويُنفذ على مدى {weeks} أسابيع تقريباً.",
        "exec_capstone": "ينجز كل مشارك مشروعاً ختامياً، فينتهي البرنامج بعمل يمكن للجهة الاستفادة منه.",
        "exec_measure": "يُقاس التقدم من خلال {assess}.",
        "participants_level": "المستوى: {level}.",
        "participants_prereq": "المتطلبات السابقة: {pre}.",
        "method_practice": "نحو {pct}% من وقت التدريب تطبيق عملي: معامل وتمارين والمشروع الختامي.",
        "method_formats": "أسلوب التنفيذ: {formats}.",
        "method_bullets": ["شرح موجز من المدرب يليه تطبيق موجه", "تمارين عملية على حالات واقعية من قطاع {ind}",
                           "اختبارات قصيرة بعد كل وحدة للتحقق من الفهم", "تغذية راجعة من المدرب على كل تكليف عملي"],
        "schedule_line": "المدة: {hours:g} ساعة على مدى {weeks} أسابيع تقريباً ({per_week} ساعات أسبوعياً).",
        "schedule_client": "تُحدد المواعيد النهائية بالاتفاق مع العميل؛ والدفعات العامة القادمة في الأكاديمية هي:",
        "invest_fee": "الرسوم لكل مشارك (الدورة العامة): {fee:,.0f} {cur}",
        "invest_group": "تُحدد أسعار المجموعات والتدريب الداخلي والمناقصات في العرض المالي.",
        "quality_review": "تمت مراجعة المنهج واعتماده داخلياً بواسطة {by} بتاريخ {date}.",
        "quality_cov": "{fw}: نسبة تغطية {pct:g}%",
        "quality_none": "لا يتوفر تقرير مواءمة مع الأطر حتى الآن.",
        "disclaimer": "المواءمة ليست اعتماداً رسمياً: تقارن هذه الأرقام محتوى الدورة بموضوعات الأطر المنشورة، ولا تعني "
                      "أن الدورة معتمدة من أي جهة.",
        "next_steps": ["تأكيد عدد المشاركين والمواعيد المفضلة", "الاتفاق على حالات أو بيانات المشروع الختامي مع المختص لدى العميل",
                       "تأكيد العرض المالي وتوقيع اتفاقية التدريب", "استلام المشاركين تعليمات الالتحاق ودليل المتدرب"],
        "f_department": "القسم", "f_audience": "الفئة المستهدفة", "f_level": "المستوى", "f_industry": "القطاع",
        "f_duration": "مدة الدورة", "f_country": "الدولة", "f_framework": "إطار الاعتماد (مواءمة فقط)",
        "f_prereq": "المتطلبات السابقة", "f_sources": "مصادر البيانات وتاريخ التحديث", "f_modules": "الوحدات",
        "f_format": "أسلوب التدريب", "f_assessments": "التقييمات", "f_version": "إصدار المنهج",
        "welcome": "مرحباً بك في دورة {title}. يرافقك هذا الدليل طوال الدورة: يوضح ما تغطيه كل وحدة، والأنشطة التي "
                   "ستنفذها، والأسئلة التي تتأمل فيها بعد كل وحدة.",
        "welcome_use": "أحضره إلى كل جلسة، ودوّن ملاحظاتك فيه، واستخدم أسئلة التأمل للاستعداد للاختبارات والمشروع الختامي.",
        "module_intro": "تستغرق هذه الوحدة نحو {h:g} ساعة، وستتعلم فيها:",
        "module_no_topics": "سيغطي المدرب {m} بالتفصيل؛ وفيما يلي النقاط الرئيسية من المناهج المرجعية.",
        "activities": "الأنشطة", "reflect": "أسئلة للتأمل", "further": "قراءات إضافية", "key_points": "نقاط رئيسية",
        "activity_from_topic": "تدرّب على {t}: تابع عرض المدرب، ثم كرره على مثال من عندك ودوّن ما غيّرته",
        "reflect_q": ["أي فكرة من وحدة \"{m}\" ستستخدمها أولاً في عملك، وكيف؟",
                      "ما أصعب جزء في هذه الوحدة، وماذا ستفعل لإتقانه؟", "كيف تشرح {t} لزميلك في جملتين؟"],
        "ref_point": "{topic}", "ref_outcome": "ناتج مرجعي: {o}", "ref_cite": "المصدر: {c}",
        "pass_rule": "للنجاح تحتاج إلى {pass_mark}% على الأقل من الدرجة الكلية ونسبة حضور {attendance}%، مع تسليم "
                     "المشروع الختامي.",
        "capstone_deliver": "ستسلّم خطة المشروع، و{artifact} يعمل، وتقريراً موجزاً، وعرضاً تقديمياً ختامياً.",
        "support_instructor": "مدربك: {name} ({title})",
        "support_general": ["اطرح أسئلتك أثناء الجلسات أو في مساحة النقاش الخاصة بالدورة",
                            "احجز ساعات مكتبية مع المدرب للمساعدة في المشروع الختامي",
                            "تواصل مع منسق التدريب في الأكاديمية لأسئلة المواعيد والشهادات"],
        "glance_modules": "{n} وحدات + المشروع الختامي",
        "default_plan_note": "خطة التقييم الافتراضية للأكاديمية: تُعتمد الأوزان النهائية في المنهج المعتمد.",
        "purpose": "يساعدك هذا الدليل على تقديم دورة {title} بشكل متسق. يتضمن خطة الجلسات، وما يلزم تحضيره، وخطة زمنية "
                   "لكل وحدة، وتعليمات الأنشطة، وأسئلة التحقق من الفهم، والصعوبات الشائعة لدى المتدربين.",
        "purpose_2": "يستخدم المتدربون دليل المتدرب؛ أما بنك الأسئلة ومعايير التقييم فهي للمدرب فقط.",
        "fac_profile": "ممارس في مجال {dept} لديه خبرة في العمل بقطاع {ind} وفي تدريب الكبار بمستوى {level}.",
        "fac_assigned": "المدربون المعينون في الكتالوج:",
        "prep_common": ["أدلة المتدربين (مطبوعة أو إلكترونية) وكشف الحضور", "بنك الأسئلة ومعايير التقييم (سرية)",
                        "الشرائح المعدة من مخطط العرض التقديمي", "مساحة على منصة التعلم للتسليمات والإعلانات"],
        "prep_tech": ["جهاز أو معمل افتراضي لكل متدرب مثبت عليه أدوات الدورة بعد اختبارها",
                      "بيانات أو مشروعات نموذجية لكل معمل مع حلولها", "نسخ احتياطية من مواد المعمل تحسباً لمشكلات الشبكة"],
        "prep_case": ["دراسات حالة ونماذج مطبوعة لكل تمرين", "لوح ورقي أو سبورة رقمية للعمل الجماعي"],
        "sessions": "الجلسات ({n} س)", "main_activities": "الأنشطة الرئيسية", "segment": "المرحلة",
        "minutes": "الدقائق", "what_to_do": "ماذا تفعل",
        "seg": {"open": "الافتتاح والمراجعة", "teach": "الشرح والعرض العملي", "practice": "التطبيق الموجه",
                "review": "المراجعة والتحقق من الفهم"},
        "seg_do": {"open": "راجع الوحدة السابقة، واذكر أهداف اليوم وأهميتها في العمل بقطاع {ind}",
                   "teach": "اشرح {topics}، وقدّم مثالاً عملياً لكل منها",
                   "practice": "ينفذ المتدربون الأنشطة في أزواج؛ تجوّل بينهم وقدم التغذية الراجعة ودوّن الأخطاء الشائعة",
                   "review": "اطرح أسئلة التحقق، ونفّذ اختبار الوحدة، وأجب عن الأسئلة، وقدّم الوحدة التالية"},
        "mod_objectives": "في نهاية هذه الوحدة يتمكن المتدربون من التعامل مع: {topics}.",
        "fac_activity": "{x} - اشرح المهمة (5 دقائق)، واترك المتدربين يعملون، ثم ناقش حلّين مع المجموعة",
        "check_q": "اسأل: \"اشرح {t} بأسلوبك وقدّم مثالاً من قطاع {ind}.\"",
        "difficulties": "صعوبات شائعة", "check": "التحقق من الفهم",
        "diff_generic": ["ينسخ المتدربون العرض العملي دون فهم السبب: اطلب منهم تغيير مُدخل واحد وتوقع النتيجة",
                         "تفاوت المستويات في المجموعة: اجمع المتقدمين مع الجدد في أزواج",
                         "تجاوز وقت التطبيق: حافظ على وقت المراجعة وانقل التمارين الإضافية إلى واجب منزلي"],
        "assess_admin": "{name} ({w:g}%): {how}",
        "assess_how": {"quiz": "في نهاية كل وحدة؛ 10-15 سؤالاً من بنك الأسئلة؛ أعلن النتائج في الجلسة التالية",
                       "exam": "اختبار مغلق محدد الوقت، أسئلته من بنك الأسئلة لجميع الوحدات",
                       "assignment": "يُكلف به في نهاية الوحدة ويُسلم قبل التالية؛ صححه بمعايير التقييم وأعد التغذية "
                                     "الراجعة خلال أسبوع",
                       "project": "عرّف المتدربين به مبكراً، وراجع الخطة قبل بدء التنفيذ، وصححه بمعايير تقييم المشروع",
                       "presentation": "10 دقائق لكل متدرب أو فريق مع الأسئلة؛ يُقيَّم بمعايير التقييم"},
        "capstone_sup": ["قدّم موجز المشروع في الأسبوع الأول ليجمع المتدربون الأفكار", "اعتمد خطة كل مشروع قبل بدء التنفيذ",
                         "اعقد لقاء متابعة قصيراً عند كل مرحلة وسجّل التقدم", "صحح التسليم النهائي والعرض بمعايير تقييم المشروع",
                         "تحقق من عدم وجود انتحال أو استخدام غير مُعلن لأدوات الذكاء الاصطناعي"],
        "slide_title": "مقدمة الدورة", "slide_agenda": "جدول الأعمال", "slide_outcomes": "نواتج التعلم",
        "slide_activity": "نشاط: {m}", "slide_capstone": "المشروع الختامي", "slide_summary": "الخلاصة والأسئلة",
        "slide_hours": "{hours:g} ساعة · {n} وحدات · {level}",
        "notes_title": "رحّب بالمجموعة، وعرّف بنفسك، واسأل كل متدرب عما يريده من الدورة.",
        "notes_agenda": "استعرض الوحدات ووضّح موقع المشروع الختامي.",
        "notes_outcomes": "اقرأ النواتج واسأل أيها الأهم لعملهم.",
        "notes_module": "قدّم الوحدة: أهميتها في العمل بقطاع {ind} وما سيتمكن المتدربون من فعله.",
        "notes_topic": "اشرح {t}؛ واعرض مثالاً عملياً من قطاع {ind}؛ واطلب من متدرب إعادة شرحه.",
        "topic_bullets": ["ما هو ولماذا هو مهم", "كيف يُستخدم في العمل بقطاع {ind}", "مثال عملي", "أخطاء شائعة يجب تجنبها"],
        "notes_activity": "اشرح النشاط وحدد الوقت، ثم ناقش حلّين.",
        "notes_capstone": "اعرض الموجز والمراحل ومعايير التقييم، وأجب عن الأسئلة حول نطاق المشروع.",
        "notes_summary": "لخّص النواتج، واجمع التغذية الراجعة، ووضّح الخطوات التالية (الشهادة، الدراسة الإضافية).",
        "summary_bullets": ["ما الذي غطيناه", "كيف سيتم تقييمك", "الخطوات التالية بعد الدورة"],
        "usage": "سري - للمدربين فقط. الأسئلة مجمعة حسب الوحدة، ويظهر لكل سؤال نوعه والمستوى المعرفي وناتج التعلم الذي "
                 "يقيسه. استخدم أسئلة الاختيار من متعدد والصواب والخطأ في اختبارات الوحدات، والأسئلة القصيرة والعملية "
                 "في التكليفات والاختبارات النهائية.",
        "usage_rules": "الأسئلة المولدة بالقواعد تقيس معرفة هيكل الدورة ونواتجها؛ ويجب أن يراجعها المدرب (ويضيف أسئلة "
                       "في المحتوى) قبل الاستخدام.",
        "usage_llm": "كُتبت الأسئلة بواسطة نموذج اللغة من المنهج والمناهج المرجعية؛ ويجب أن يراجع المدرب كل سؤال قبل "
                     "الاستخدام.",
        "qt": {"mcq": "اختيار من متعدد", "tf": "صواب / خطأ", "short": "إجابة قصيرة", "practical": "عملي"},
        "q_cols": ["#", "النوع", "المستوى", "الناتج", "السؤال", "الخيارات", "الإجابة / عناصر التصحيح"],
        "q_mcq": "أي مما يلي من موضوعات وحدة \"{m}\"؟",
        "q_mcq2": "تغطي وحدة \"{m}\" أيضاً أياً من الموضوعات التالية؟",
        "q_tf": "\"{t}\" جزء من وحدة \"{m}\".",
        "q_short": "اشرح {t} وقدّم مثالاً من العمل في قطاع {ind}.",
        "q_short_ans": "شرح صحيح لـ {t} (2)؛ مثال مناسب (1)",
        "q_practical_ans": "اتباع الخطوات (2)؛ نتيجة صحيحة (2)؛ تبرير الاختيارات (1)",
        "q_ref_ans": "عناصر التصحيح من المنهج المرجعي ({c})",
        "true": "صواب", "false": "خطأ",
        "no_items": "موضوعات هذه الوحدة غير كافية لتوليد أسئلة - أضف موضوعات إلى المنهج أو اكتب الأسئلة يدوياً.",
        "bp_cols": ["الوحدة", "اختيار من متعدد", "صواب / خطأ", "إجابة قصيرة", "عملي", "الإجمالي"],
        "levels": [("ممتاز", 4, "يحقق المعيار بالكامل دون أخطاء مع فهم عميق"), ("جيد", 3, "يحقق المعيار مع أخطاء بسيطة"),
                   ("مقبول", 2, "يحقق المعيار جزئياً مع عدة أخطاء أو نواقص"), ("يحتاج إلى تحسين", 1, "لم يحقق المعيار بعد")],
        "scale_cols": ["المستوى", "النقاط", "المعنى"],
        "rubric_intro": "الوزن: {w:g}% من الدرجة النهائية ({pts:g} نقطة). النواتج المقيسة: {los}.",
        "rubric_cols": ["المعيار (النقاط)", "ممتاز (4)", "جيد (3)", "مقبول (2)", "يحتاج إلى تحسين (1)"],
        "crit": {
            "project": [("المتطلبات والنطاق", "جميع المتطلبات محققة والنطاق محدد بوضوح", "معظم المتطلبات محققة",
                         "عدة متطلبات ناقصة", "معظم المتطلبات غير محققة"),
                        ("التطبيق الصحيح لمفاهيم الدورة", "تطبيق صحيح ومبرر", "أخطاء بسيطة في التطبيق",
                         "عدة أخطاء وتبرير محدود", "تطبيق خاطئ"),
                        ("جودة المخرج", "مكتمل ويعمل ومنظم", "يعمل مع مشكلات بسيطة", "يعمل جزئياً أو غير مكتمل", "لا يعمل"),
                        ("التوثيق والعرض", "واضح وموجز مع إجابات واثقة", "واضح مع نواقص بسيطة", "صعب المتابعة أحياناً",
                         "غير واضح أو غير موجود")],
            "assignment": [("صحة النتائج", "جميع النتائج صحيحة", "أخطاء بسيطة", "عدة أخطاء", "معظمها خاطئ"),
                           ("استخدام الأساليب والأدوات", "أساليب مناسبة مستخدمة بكفاءة", "مناسبة مع ملاحظات بسيطة",
                            "مناسبة جزئياً", "غير مناسبة"),
                           ("التحليل والتفسير", "عميق ومدعوم", "سليم", "سطحي", "غير موجود"),
                           ("وضوح التسليم", "منظم وسهل المتابعة", "واضح في معظمه", "صعب المتابعة", "غير منظم")],
            "presentation": [("المحتوى", "دقيق ومكتمل", "نواقص بسيطة", "عدة نواقص", "غير دقيق"),
                             ("التنظيم والوقت", "منطقي وفي الوقت المحدد", "منطقي في معظمه", "صعب المتابعة", "غير منظم"),
                             ("الإجابة عن الأسئلة", "واثقة وصحيحة", "صحيحة في معظمها", "جزئية", "لا يستطيع الإجابة")],
            "score": [("الدرجة", "85-100%", "70-84%", "{pm}-69%", "أقل من {pm}%")],
        },
        "grading": ["درجة النجاح: {pm}% من مجموع الدرجات الموزونة", "الحد الأدنى للحضور: {att}%",
                    "يجب تسليم المشروع الختامي للنجاح", "أوزان التقييم: {weights}"],
        "scenario": "تعمل في جهة من قطاع {ind} في {country}. {need} مهمتك أن تخطط لحل يطبق ما تعلمته في هذه الدورة "
                    "وتنفذه وتعرضه.",
        "req_module": "تطبيق {m}{detail}",
        "req_general": ["توثيق الافتراضات ومصادر البيانات وقرارات التصميم", "عرض النتيجة على جمهور غير متخصص في 10 دقائق"],
        "deliverables": ["خطة المشروع (النطاق، المنهجية، الجدول الزمني) - يعتمدها المدرب", "{artifact} يعمل",
                         "تقرير موجز (3-5 صفحات): المشكلة، المنهجية، النتائج، التوصيات", "عرض تقديمي ختامي مع عرض عملي"],
        "ms_cols": ["المرحلة", "أسبوع الدورة", "الساعات", "المخرج"],
        "ms": [("التخطيط", "اعتماد خطة المشروع"), ("التنفيذ", "{artifact} يعمل"),
               ("المراجعة", "تطبيق ملاحظات الزملاء والمدرب"), ("العرض", "التقرير والعرض التقديمي")],
        "submission": ["سلّم جميع الملفات على منصة التعلم قبل الجلسة الأخيرة", "سمِّ الملفات: <الدورة>_<اسمك>_<المخرج>",
                       "اعرض مشروعك في الجلسة الأخيرة: 10 دقائق و5 دقائق للأسئلة", "التسليم المتأخر يحتاج موافقة المدرب"],
        "integrity": ["يجب أن يكون العمل المقدم من إنجازك (أو إنجاز فريقك في المشروعات الجماعية)",
                      "اذكر مصدر كل مرجع أو بيانات أو شيفرة تعيد استخدامها",
                      "يمكن لأدوات الذكاء الاصطناعي مساعدتك في التعلم والمسودات، لكن يجب الإفصاح عن طريقة استخدامها "
                      "والقدرة على شرح كل جزء من عملك",
                      "الانتحال أو العمل المولد بالذكاء الاصطناعي دون إفصاح يؤدي إلى الرسوب في المشروع الختامي"],
        "capstone_default": "مشروع ختامي يطبق فيه المتدربون وحدات دورة {title} على حالة واقعية من قطاع {ind}.",
        "plan_intro": "توضح هذه الخطة كيف يُدرَّس كل ناتج تعلم في دورة {title} وكيف يُقيَّم، بحيث يُقاس كل ناتج مرة واحدة "
                      "على الأقل ويبلغ مجموع الأوزان 100%.",
        "map_cols": ["الناتج", "ناتج التعلم", "المستوى", "يُدرَّس في", "يُقيَّم بواسطة"],
        "as_cols": ["التقييم", "النوع", "الوزن", "النواتج المقيسة"],
        "cov_ok": "كل ناتج تعلم يُقيَّم مرة واحدة على الأقل.",
        "cov_missing": "غير مُقيَّم بعد: الناتج {n} \"{o}\" - أضفه إلى أحد التقييمات.",
        "cov_verb": "الناتج {n} لا يبدأ بفعل قابل للقياس - أعد صياغته (مثل \"يشرح ...\"، \"يبني ...\").",
        "cov_weights": "مجموع أوزان التقييم {w:g}% (يجب أن يكون 100%).",
        "cov_weights_ok": "مجموع أوزان التقييم 100%.",
        "cov_all_modules": "جميع الوحدات",
    },
}
TYPE_AR = {"quiz": "اختبار قصير", "exam": "اختبار نهائي", "assignment": "تكليف", "project": "مشروع",
           "presentation": "عرض تقديمي", "lab": "معمل"}


def bloom_level(outcome: str) -> str:
    words = _lead_words(outcome)
    norm = {lvl: {normalize(v) for v in vs} for lvl, vs in BLOOM_VERBS.items()}
    for w in words:
        for lvl, vs in norm.items():
            if w in vs:
                return lvl
    return "understand"


def outcome_modules(outcome: str, modules: list[tuple[int, dict]]) -> list[int]:
    """Modules whose title or topics share a distinctive word with the outcome."""
    ow = _words(outcome)
    return [n for n, m in modules if ow & _words(f"{m['title']} {' '.join(m.get('topics') or [])}")]


def outcome_assessments(outcome: str, assessments: list[dict]) -> list[str]:
    domains = outcome_domain(outcome) or {"knowledge"}
    types = set().union(*(ASSESSES.get(d, set()) for d in domains))
    return [a["name"] for a in assessments if (a.get("type") or "").lower() in types]


class Builder:
    def __init__(self, facts: CourseFacts, template: tpl.Template, lang: str, use_references: bool = True):
        self.f = facts
        self.t_ = template
        self.lang = lang if lang in P else "en"
        self.L = P[self.lang]
        self.doc_type = template.doc_type
        self.use_references = use_references
        self.meta = tpl.meta()

    # ------------------------------------------------------------------ helpers
    def t(self, key: str, **fmt) -> str:
        text = self.L[key]
        return text.format(**fmt) if fmt else text

    def sec(self, s: tpl.Section, key: Optional[str] = None, heading: Optional[str] = None, **kw) -> DocSection:
        return DocSection(key=key or s.key, heading=heading or s.title(self.lang), locked=s.locked, **kw)

    def table(self, columns: list[str], rows: list[list], widths: Optional[list[float]] = None) -> DocTable:
        return DocTable(columns=columns, rows=[[self._cell(c) for c in r] for r in rows], widths=widths)

    @staticmethod
    def _cell(v) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    def mid(self, text: str) -> str:
        """English text placed mid-sentence starts lower-case ("Explain the seven testing principles"); acronyms and
        names such as "SQL" or "Power BI" stay as they are."""
        if self.lang != "en" or not text:
            return text
        first = text.split()[0]
        if len(first) > 1 and (first.isupper() or first[1:] != first[1:].lower()):
            return text
        return text[0].lower() + text[1:]

    def mids(self, items: list[str]) -> str:
        return ("، " if self.lang == "ar" else ", ").join(self.mid(x) for x in items)

    @property
    def ind(self) -> str:
        ind = self.f.industry or ("workplace" if self.lang == "en" else "العمل")
        return INDUSTRY_AR.get(ind, ind) if self.lang == "ar" else self.mid(ind)

    @property
    def dept(self) -> str:
        d = self.f.department or "-"
        return DEPARTMENT_AR.get(d, d) if self.lang == "ar" else d

    @property
    def level(self) -> str:
        lv = self.f.level or "-"
        return LEVEL_AR.get(lv, lv) if self.lang == "ar" else lv.lower()

    @property
    def country(self) -> str:
        return COUNTRIES[self.lang].get(self.f.country or "", self.f.country or "-")

    @property
    def artifact(self) -> str:
        if self.lang == "ar":
            return ARTIFACT_AR.get(self.f.department or "", "حل")
        return ARTIFACT.get(self.f.department or "", "solution")

    @property
    def per_week(self) -> int:
        return min(HOURS_PER_WEEK, math.ceil(self.f.hours / max(self.f.weeks, 1))) if self.f.hours else 0

    def type_name(self, t: str) -> str:
        return TYPE_AR.get(t, t) if self.lang == "ar" else t

    def fmt_name(self, f) -> str:
        return self.L["formats"].get(getattr(f, "value", f), str(f))

    def topics(self, n: int, mod: dict) -> list[str]:
        """Module topics; for catalogue modules (no topics) the topics of the matching reference syllabus."""
        if mod.get("topics"):
            return list(mod["topics"])
        refs = self.f.module_refs.get(n) or []
        return list(refs[0].topics) if refs else []

    def first_topic(self, n: int, mod: dict) -> str:
        tops = self.topics(n, mod)
        return tops[0] if tops else mod["title"]

    def assessments_text(self) -> str:
        sep = "، " if self.lang == "ar" else ", "
        return sep.join(a["name"] for a in self.f.assessments) or self.L["tbc"]

    def outcome_label(self, i: int) -> str:
        return f"LO{i}" if self.lang == "en" else f"ن{i}"

    def lo_for(self, n: int, mod: dict) -> str:
        """Best-matching outcome for a module (most shared words), as LOn."""
        mw = _words(f"{mod['title']} {' '.join(self.topics(n, mod))}")
        scored = [(len(mw & _words(o)), i) for i, o in enumerate(self.f.outcomes, start=1)]
        best = max(scored, default=(0, 0))
        return self.outcome_label(best[1]) if best[0] else "-"

    # ------------------------------------------------------------------ build
    def build(self) -> list[DocSection]:
        out: list[DocSection] = []
        for s in self.t_.sections:
            fn = getattr(self, f"s_{self.doc_type}__{s.key}", None) or getattr(self, f"s_{s.key}", None)
            if fn is None:
                log.warning("Template %s: no builder for section '%s' - skipped", self.t_.template_id, s.key)
                continue
            res: Union[DocSection, list[DocSection], None] = fn(s)
            if isinstance(res, DocSection):
                out.append(res)
            elif res:
                out.extend(res)
        return out

    # ------------------------------------------------------------------ shared sections
    def s_objectives(self, s):
        return self.sec(s, bullets=self.f.objectives or [self.L["tbc"]])

    def s_outcomes(self, s):
        intro = self.L["outcomes_intro_formal" if self.doc_type in ("course_proposal", "curriculum_pdf",
                                                                     "capstone_brief") else "outcomes_intro"]
        outcomes = self.f.outcomes
        if self.doc_type == "capstone_brief":
            chosen = [o for o in outcomes if outcome_domain(o) & {"practical", "intellectual", "transferable"}]
            outcomes = chosen or outcomes
        return self.sec(s, paragraphs=[intro], bullets=outcomes or [self.L["tbc"]], numbered=True)

    def s_audience(self, s):
        return self.sec(s, paragraphs=[self.f.audience, self.t("level_line", level=self.level)])

    def s_prerequisites(self, s):
        return self.sec(s, bullets=self.f.prerequisites or [self.L["none"]])

    def s_capstone(self, s):
        paras = [self.f.capstone or self.t("capstone_default", title=self.f.title, ind=self.ind)]
        if self.doc_type == "participant_guide":
            paras.append(self.t("capstone_deliver", artifact=self.artifact))
        return self.sec(s, paragraphs=paras)

    def s_assessment_table(self, s):
        L = self.L
        rows = [[a["name"], self.type_name(a["type"]), f"{float(a['weight_pct']):g}%"] for a in self.f.assessments]
        rows.append([L["total"], "", f"{sum(float(a['weight_pct']) for a in self.f.assessments):g}%"])
        cols = [L["assessment"], L["type"], L["weight"]]
        widths = [4, 1.5, 1]
        if self.doc_type == "outcomes_assessment_plan":
            cols = L["as_cols"]
            widths = [3, 1.3, 0.9, 2]
            for r, a in zip(rows, self.f.assessments):
                los = [self.outcome_label(i) for i, o in enumerate(self.f.outcomes, 1)
                       if a["name"] in outcome_assessments(o, self.f.assessments)]
                r.append(", ".join(los) or "-")
            rows[-1].append("")
        paras = [L["default_plan_note"]] if self.f.default_assessments else []
        return self.sec(s, paragraphs=paras, table=self.table(cols, rows, widths))

    def s_alignment(self, s):
        L = self.L
        if not self.f.alignment:
            return self.sec(s, paragraphs=[L["quality_none"]])
        rows = []
        for r in self.f.alignment:
            gaps = (r.get("missing_topics") or []) + (r.get("missing_outcomes") or [])
            rows.append([r["framework"], f"{float(r['coverage_pct']):g}%", "; ".join(gaps[:4]) or L["none"]])
        return self.sec(s, table=self.table([L["framework"], L["coverage"], L["gaps"]], rows, [2.2, 0.9, 3.5]),
                        paragraphs=[L["disclaimer"]])

    def s_grading(self, s):
        sep = "، " if self.lang == "ar" else ", "
        weights = sep.join(f"{a['name']} {float(a['weight_pct']):g}%" for a in self.f.assessments)
        bullets = [x.format(pm=self.meta.get("pass_mark_pct", 60), att=self.meta.get("attendance_pct", 80),
                            weights=weights or self.L["tbc"]) for x in self.L["grading"]]
        return self.sec(s, bullets=bullets)

    def _instructor_lines(self) -> list[str]:
        return [self.t("instructor", name=i.name, title=i.title, years=i.years_experience,
                       bio=i.bio_ar if self.lang == "ar" else i.bio_en) for i in self.f.instructors]

    def _cohort_lines(self) -> list[str]:
        out = []
        for c in self.f.schedules[:4]:
            loc = f", {c.location}" if c.location else ""
            out.append(self.t("cohort", date=c.start_date.isoformat(), fmt=self.fmt_name(c.format), days=c.days,
                              time=c.time, loc=loc, seats=c.seats_left))
        return out

    def _fee_lines(self) -> list[str]:
        c = self.f.course
        if not c or c.fee is None:
            return [self.L["no_fee"]]
        out = [self.t("fee", fee=c.fee, cur=c.currency)]
        for p in self.f.promotions:
            disc_price = round(c.fee * (1 - p.discount_pct / 100))
            out.append(self.t("fee_promo", name=p.name_ar if self.lang == "ar" else p.name_en, disc=p.discount_pct,
                              cond=p.condition_ar if self.lang == "ar" else p.condition_en,
                              until=p.valid_until.isoformat(), price=disc_price, cur=c.currency))
        if c.installments:
            out.append(self.L["installments"])
        return out

    def _structure_rows(self, with_topics: bool = True, with_practice: bool = False) -> list[list]:
        rows = []
        for n, m in enumerate(self.f.modules, start=1):
            row = [n, m["title"]]
            if with_topics:
                row.append("; ".join(self.topics(n, m)) or "-")
            if with_practice:
                row.append("; ".join(m.get("exercises") or []) or "-")
            row.append(f"{float(m.get('hours') or 0):g}")
            rows.append(row)
        total = sum(float(m.get("hours") or 0) for m in self.f.modules)
        rows.append(["", self.L["total"]] + [""] * (int(with_topics) + int(with_practice)) + [f"{total:g}"])
        return rows

    # ------------------------------------------------------------------ course description
    def s_course_description__overview(self, s):
        c = self.f.course
        desc = (c.description_ar if self.lang == "ar" else c.description_en) if c else ""
        paras = [desc] if desc else []
        if not desc or self.f.origin == "curriculum":
            paras.insert(0, self.t("desc_compose", title=self.f.title, hours=self.f.hours, level=self.level,
                                   dept=self.dept, aud=self.mid(self.f.audience)))
            if self.f.objectives and self.f.origin == "curriculum":
                aim = self.f.objectives[0].rstrip(".")
                paras.append(self.t("desc_aim", aim=aim[0].lower() + aim[1:] if self.lang == "en" else aim))
        return self.sec(s, paragraphs=paras)

    def s_content_table(self, s):
        L = self.L
        return self.sec(s, table=self.table(["#", L["module"], L["hours"]], self._structure_rows(with_topics=False),
                                            [0.5, 5, 1]))

    def s_format(self, s):
        c = self.f.course
        bullets = [self.t("duration", hours=self.f.hours, weeks=self.f.weeks, per_week=self.per_week)]
        if c:
            sep = "، " if self.lang == "ar" else ", "
            bullets.append(self.t("formats_line", formats=sep.join(self.fmt_name(x) for x in c.formats)))
            names = {"en": "English" if self.lang == "en" else "الإنجليزية", "ar": "Arabic" if self.lang == "en"
                     else "العربية"}
            bullets.append(self.t("language_line", langs=sep.join(names.get(x.value, x.value) for x in c.language)))
        cohorts = self._cohort_lines()
        if cohorts:
            bullets.append(self.L["next_cohorts"])
            bullets += cohorts
        else:
            bullets.append(self.L["no_cohort"])
        return self.sec(s, bullets=bullets)

    def s_instructors(self, s):
        return self.sec(s, bullets=self._instructor_lines() or [self.L["no_instructor"]])

    def s_fees(self, s):
        return self.sec(s, bullets=self._fee_lines())

    def s_certificate(self, s):
        c = self.f.course
        paras = [c.certificate] if c and c.certificate else []
        if self.f.framework:
            paras.append(self.t("certificate_align", fw=self.f.framework))
        return self.sec(s, paragraphs=paras or [self.L["tbc"]])

    def s_careers(self, s):
        c = self.f.course
        return self.sec(s, bullets=list(c.career_opportunities) if c and c.career_opportunities
                        else [self.L["no_careers"]])

    # ------------------------------------------------------------------ proposal
    def client(self) -> str:
        return self._client or self.L["client_default"]

    _client: Optional[str] = None

    def s_prepared_for(self, s):
        from datetime import date
        return self.sec(s, paragraphs=[self._client or self.L["client_default"], self.L["prepared_by"],
                                       self.t("date_line", date=date.today().isoformat())])

    def s_executive_summary(self, s):
        n = len(self.f.teaching_modules)
        paras = [self.t("exec_need", client=self.client(), aud=self.mid(self.f.audience), topic=self.mid(self.f.title)),
                 self.t("exec_offer", title=self.f.title, hours=self.f.hours, level=self.level, n=n,
                        weeks=self.f.weeks),
                 self.L["exec_capstone"], self.t("exec_measure", assess=self.assessments_text())]
        return self.sec(s, paragraphs=[" ".join(paras)])

    def s_participants(self, s):
        sep = "؛ " if self.lang == "ar" else "; "
        return self.sec(s, paragraphs=[self.f.audience, self.t("participants_level", level=self.level),
                                       self.t("participants_prereq", pre=sep.join(self.f.prerequisites) or self.L["none"])])

    def s_program_table(self, s):
        L = self.L
        return self.sec(s, table=self.table(["#", L["module"], L["topics"], L["hours"]], self._structure_rows(),
                                            [0.4, 2.2, 4, 0.8]))

    def practice_pct(self) -> int:
        if not self.f.hours:
            return 0
        cap = sum(float(m.get("hours") or 0) for n, m in enumerate(self.f.modules, 1)
                  if (n, m) not in self.f.teaching_modules)
        labs = sum(float(m.get("hours") or 0) * 0.4 for _, m in self.f.teaching_modules)
        return int(round(100 * (cap + labs) / self.f.hours))

    def s_methodology(self, s):
        c = self.f.course
        sep = "، " if self.lang == "ar" else ", "
        paras = [self.t("method_practice", pct=self.practice_pct())]
        if c:
            paras.append(self.t("method_formats", formats=sep.join(self.fmt_name(x) for x in c.formats)))
        return self.sec(s, paragraphs=paras, bullets=[b.format(ind=self.ind) for b in self.L["method_bullets"]])

    def s_team(self, s):
        return self.sec(s, bullets=self._instructor_lines() or [self.L["no_instructor"]])

    def s_schedule(self, s):
        paras = [self.t("schedule_line", hours=self.f.hours, weeks=self.f.weeks, per_week=self.per_week)]
        cohorts = self._cohort_lines()
        if cohorts:
            paras.append(self.L["schedule_client"])
        return self.sec(s, paragraphs=paras, bullets=cohorts)

    def s_investment(self, s):
        c = self.f.course
        bullets = [self.t("invest_fee", fee=c.fee, cur=c.currency)] if c and c.fee is not None else [self.L["no_fee"]]
        if c and c.fee is not None:
            bullets += self._fee_lines()[1:]
        bullets.append(self.L["invest_group"])
        return self.sec(s, bullets=bullets)

    def s_quality(self, s):
        L = self.L
        paras = []
        if self.f.curriculum_status == "approved" and self.f.approved_by:
            paras.append(self.t("quality_review", by=self.f.approved_by, date=(self.f.approved_at or "")[:10]))
        bullets = [self.t("quality_cov", fw=r["framework"], pct=float(r["coverage_pct"])) for r in self.f.alignment]
        paras.append(L["disclaimer"] if bullets else L["quality_none"])
        return self.sec(s, paragraphs=paras, bullets=bullets)

    def s_next_steps(self, s):
        return self.sec(s, bullets=list(self.L["next_steps"]), numbered=True)

    # ------------------------------------------------------------------ curriculum
    def s_info_table(self, s):
        L, f = self.L, self.f
        sources = ", ".join(dict.fromkeys(x.source_id for x in f.sources)) or L["none"]
        rows = [[L["f_department"], self.dept], [L["f_audience"], f.audience], [L["f_level"], self.level],
                [L["f_industry"], self.ind], [L["f_duration"], f"{f.hours:g} {L['hours']} · {f.weeks} {L['weeks']}"],
                [L["f_country"], self.country], [L["f_framework"], f.framework or L["none"]],
                [L["f_prereq"], "; ".join(f.prerequisites) or L["none"]], [L["f_sources"], sources]]
        if f.curriculum_version:
            rows.append([L["f_version"], f"v{f.curriculum_version} ({f.curriculum_status})"])
        return self.sec(s, table=self.table([L["field"], L["value"]], rows, [1.6, 4.4]))

    def s_structure_table(self, s):
        L = self.L
        return self.sec(s, table=self.table(["#", L["module"], L["topics"], L["practical"], L["hours"]],
                                            self._structure_rows(with_practice=True), [0.4, 1.8, 3, 2.2, 0.7]))

    def s_exercises(self, s):
        module_ex = {e for m in self.f.modules for e in m.get("exercises") or []}
        extra = [e for e in self.f.exercises if e not in module_ex] or sorted(module_ex)
        return self.sec(s, bullets=extra or [self.L["none"]])

    # ------------------------------------------------------------------ participant guide
    def s_welcome(self, s):
        return self.sec(s, paragraphs=[self.t("welcome", title=self.f.title), self.L["welcome_use"]])

    def s_glance_table(self, s):
        L, f, c = self.L, self.f, self.f.course
        sep = "، " if self.lang == "ar" else ", "
        rows = [[L["f_duration"], f"{f.hours:g} {L['hours']} · {f.weeks} {L['weeks']}"],
                [L["f_modules"], self.t("glance_modules", n=len(f.teaching_modules))],
                [L["f_level"], self.level], [L["f_prereq"], "; ".join(f.prerequisites) or L["none"]],
                [L["f_assessments"], self.assessments_text()]]
        if c:
            rows.insert(2, [L["f_format"], sep.join(self.fmt_name(x) for x in c.formats)])
        return self.sec(s, table=self.table([L["field"], L["value"]], rows, [1.6, 4.4]))

    def s_how_assessed(self, s):
        sec = self.s_assessment_table(s)
        sec.paragraphs.append(self.t("pass_rule", pass_mark=self.meta.get("pass_mark_pct", 60),
                                     attendance=self.meta.get("attendance_pct", 80)))
        return sec

    def _ref_bullets(self, n: int) -> tuple[list[str], list[str]]:
        """(key points, further reading) from the reference syllabi matched to module n."""
        refs = self.f.module_refs.get(n) or []
        points, reading = [], []
        for r in refs:
            if r.outcome:
                points.append(self.t("ref_outcome", o=r.outcome))
            reading.append(r.citation)
        return points, list(dict.fromkeys(reading))

    def s_module_chapters(self, s):
        out = []
        for n, m in self.f.teaching_modules:
            tops = self.topics(n, m)
            paras = [self.t("module_intro", h=float(m.get("hours") or 0))] if tops else \
                [self.t("module_no_topics", m=m["title"])]
            points, reading = self._ref_bullets(n)
            out.append(self.sec(s, key=f"{s.key}:{n}", heading=s.title(self.lang, n=n, title=m["title"]),
                                paragraphs=paras, bullets=tops, notes=points))
            acts = list(m.get("exercises") or []) or [self.t("activity_from_topic", t=self.mid(t)) for t in tops[:2]]
            out.append(DocSection(key=f"{s.key}:{n}:activities", heading=self.L["activities"], level=2,
                                  bullets=acts, numbered=True))
            first = self.first_topic(n, m)
            out.append(DocSection(key=f"{s.key}:{n}:reflect", heading=self.L["reflect"], level=2,
                                  bullets=[q.format(m=m["title"], t=self.mid(first)) for q in self.L["reflect_q"]]))
            if reading:
                out.append(DocSection(key=f"{s.key}:{n}:reading", heading=self.L["further"], level=2,
                                      bullets=reading))
        return out

    def s_support(self, s):
        bullets = [self.t("support_instructor", name=i.name, title=i.title) for i in self.f.instructors]
        return self.sec(s, bullets=bullets + list(self.L["support_general"]))

    # ------------------------------------------------------------------ facilitator guide
    def s_purpose(self, s):
        return self.sec(s, paragraphs=[self.t("purpose", title=self.f.title), self.L["purpose_2"]])

    def s_facilitators(self, s):
        lines = self._instructor_lines()
        paras = [self.t("fac_profile", dept=self.dept, ind=self.ind, level=self.level)]
        if lines:
            paras.append(self.L["fac_assigned"])
        return self.sec(s, paragraphs=paras, bullets=lines)

    def s_preparation(self, s):
        extra = self.L["prep_tech"] if self.f.department in TECH else self.L["prep_case"]
        return self.sec(s, bullets=list(self.L["prep_common"]) + list(extra))

    def s_session_plan(self, s):
        L = self.L
        rows = []
        for n, m in enumerate(self.f.modules, start=1):
            h = float(m.get("hours") or 0)
            acts = "; ".join((m.get("exercises") or [])[:2]) or "; ".join(self.topics(n, m)[:2]) or "-"
            rows.append([n, m["title"], f"{h:g}", max(1, math.ceil(h / SESSION_HOURS)) if h else "-", acts])
        rows.append(["", L["total"], f"{self.f.hours:g}", "", ""])
        cols = ["#", L["module"], L["hours"], self.t("sessions", n=SESSION_HOURS), L["main_activities"]]
        return self.sec(s, table=self.table(cols, rows, [0.4, 2.4, 0.7, 0.9, 3.2]))

    def s_module_notes(self, s):
        L = self.L
        out = []
        for n, m in self.f.teaching_modules:
            tops = self.topics(n, m)
            minutes = float(m.get("hours") or 0) * 60
            rows = [[L["seg"][seg], int(round(minutes * share / 5) * 5) if minutes else "-",
                     L["seg_do"][seg].format(ind=self.ind, topics=self.mids(tops[:4]) or m["title"])]
                    for seg, share in TIMING]
            points, reading = self._ref_bullets(n)
            notes = points + [self.t("ref_cite", c=c) for c in reading]
            out.append(self.sec(s, key=f"{s.key}:{n}", heading=s.title(self.lang, n=n, title=m["title"]),
                                paragraphs=[self.t("mod_objectives", topics=self.mids(tops) or m["title"])],
                                table=self.table([L["segment"], L["minutes"], L["what_to_do"]], rows, [1.6, 0.8, 4]),
                                notes=notes))
            acts = list(m.get("exercises") or []) or [self.t("activity_from_topic", t=self.mid(t)) for t in tops[:2]]
            out.append(DocSection(key=f"{s.key}:{n}:activities", heading=L["activities"], level=2,
                                  bullets=[self.t("fac_activity", x=a) for a in acts]))
            out.append(DocSection(key=f"{s.key}:{n}:check", heading=L["check"], level=2,
                                  bullets=[self.t("check_q", t=self.mid(t), ind=self.ind)
                                           for t in (tops[:3] or [m["title"]])]))
            out.append(DocSection(key=f"{s.key}:{n}:difficulties", heading=L["difficulties"], level=2,
                                  bullets=list(L["diff_generic"])))
        return out

    def s_assessment_admin(self, s):
        how = self.L["assess_how"]
        bullets = [self.t("assess_admin", name=a["name"], w=float(a["weight_pct"]),
                          how=how.get((a.get("type") or "").lower(), how["assignment"])) for a in self.f.assessments]
        return self.sec(s, bullets=bullets or [self.L["tbc"]])

    def s_capstone_supervision(self, s):
        return self.sec(s, paragraphs=[self.f.capstone] if self.f.capstone else [],
                        bullets=list(self.L["capstone_sup"]))

    # ------------------------------------------------------------------ presentation outline
    def s_slides(self, s):
        L, f = self.L, self.f
        slides: list[tuple[str, list[str], str, int]] = [
            (f.title, [L["slide_title"], f.audience, self.t("slide_hours", hours=f.hours, n=len(f.teaching_modules),
                                                             level=self.level)], L["notes_title"], 1),
            (L["slide_agenda"], [m["title"] for m in f.modules], L["notes_agenda"], 2),
            (L["slide_outcomes"], f.outcomes[:6], L["notes_outcomes"], 2)]
        for n, m in f.teaching_modules:
            tops = self.topics(n, m)
            slides.append((f"{L['module']} {n}: {m['title']}", tops[:5] or [m["title"]],
                           self.t("notes_module", ind=self.ind), 1))
            for t in tops[:4]:
                slides.append((t, [b.format(ind=self.ind) for b in L["topic_bullets"]],
                               self.t("notes_topic", t=self.mid(t), ind=self.ind), 2))
            acts = (m.get("exercises") or [])[:2]
            if acts:
                slides.append((self.t("slide_activity", m=m["title"]), acts, L["notes_activity"], 2))
        slides.append((L["slide_capstone"], [f.capstone or self.t("capstone_default", title=f.title, ind=self.ind)],
                       L["notes_capstone"], 2))
        slides.append((L["slide_summary"], list(L["summary_bullets"]), L["notes_summary"], 2))
        return [self.sec(s, key=f"slide:{i}", heading=s.title(self.lang, n=i, title=title), bullets=bullets,
                         notes=[notes], level=level) for i, (title, bullets, notes, level) in enumerate(slides, 1)]

    # ------------------------------------------------------------------ question bank
    def s_usage(self, s):
        return self.sec(s, paragraphs=[self.L["usage"], self.L["usage_rules"]])

    def _question_rows(self, n: int, mod: dict, pool: list[str]) -> tuple[list[list], dict[str, int]]:
        """Deterministic items for one module. pool = topics of the OTHER modules (MCQ distractors, false items)."""
        L = self.L
        tops = self.topics(n, mod)
        lo = self.lo_for(n, mod)
        bl = L["bloom"]
        rows: list[list] = []
        counts = {"mcq": 0, "tf": 0, "short": 0, "practical": 0}
        letters = LETTERS[self.lang]
        for j, topic in enumerate(tops[:2]):
            distractors = [p for p in pool if p not in tops][j * 3:(j * 3) + 3]
            if len(distractors) < 3:
                distractors = [p for p in pool if p not in tops][:3]
            if len(distractors) < 3:
                break
            pos = (n + j) % 4
            options = distractors[:pos] + [topic] + distractors[pos:3]
            rows.append([L["qt"]["mcq"], bl["remember"], lo, self.t("q_mcq" if j == 0 else "q_mcq2", m=mod["title"]),
                         "\n".join(f"{letters[k]}. {o}" for k, o in enumerate(options)), letters[pos]])
            counts["mcq"] += 1
        if tops:
            other = [p for p in pool if p not in tops]
            truthful = n % 2 == 1 or not other
            stmt = tops[-1] if truthful else other[n % len(other)]
            rows.append([L["qt"]["tf"], bl["remember"], lo, self.t("q_tf", t=stmt, m=mod["title"]),
                         f"{L['true']} / {L['false']}", L["true"] if truthful else L["false"]])
            counts["tf"] += 1
        for t in tops[:2]:
            rows.append([L["qt"]["short"], bl["understand"], lo, self.t("q_short", t=self.mid(t), ind=self.ind), "-",
                         self.t("q_short_ans", t=self.mid(t))])
            counts["short"] += 1
        for x in (mod.get("exercises") or [])[:1]:
            rows.append([L["qt"]["practical"], bl["apply"], lo, x, "-", L["q_practical_ans"]])
            counts["practical"] += 1
        for r in (self.f.module_refs.get(n) or [])[:1]:
            if r.outcome:
                rows.append([L["qt"]["practical"], bl[bloom_level(r.outcome)], lo, r.outcome, "-",
                             self.t("q_ref_ans", c=r.doc or r.citation)])
                counts["practical"] += 1
        return rows, counts

    def s_module_questions(self, s):
        L = self.L
        out, self._blueprint = [], []
        all_topics = {n: self.topics(n, m) for n, m in self.f.teaching_modules}
        q = 0
        for n, m in self.f.teaching_modules:
            pool = [t for k, ts in all_topics.items() if k != n for t in ts] or \
                   [mm["title"] for k, mm in self.f.teaching_modules if k != n]
            rows, counts = self._question_rows(n, m, pool)
            numbered = []
            for r in rows:
                q += 1
                numbered.append([q, *r])
            self._blueprint.append([m["title"], counts["mcq"], counts["tf"], counts["short"], counts["practical"],
                                    sum(counts.values())])
            heading = s.title(self.lang, n=n, title=m["title"])
            if numbered:
                out.append(self.sec(s, key=f"{s.key}:{n}", heading=heading,
                                    table=self.table(L["q_cols"], numbered, [0.35, 0.95, 1.1, 0.5, 2.5, 2.4, 1.5])))
            else:
                out.append(self.sec(s, key=f"{s.key}:{n}", heading=heading, paragraphs=[L["no_items"]]))
        return out

    def s_blueprint(self, s):
        L = self.L
        rows = list(getattr(self, "_blueprint", []))
        if rows:
            rows.append([L["total"]] + [sum(r[i] for r in rows) for i in range(1, 6)])
        return self.sec(s, table=self.table(L["bp_cols"], rows, [3, 1, 1, 1, 1, 0.8]))

    # ------------------------------------------------------------------ rubrics
    def s_scale(self, s):
        L = self.L
        return self.sec(s, table=self.table(L["scale_cols"], [[a, b, c] for a, b, c in L["levels"]], [1.4, 0.7, 4]))

    def _criteria(self, kind: str) -> list[tuple]:
        crit = self.L["crit"]
        if kind in ("quiz", "exam"):
            pm = self.meta.get("pass_mark_pct", 60)
            return [tuple(x.format(pm=pm) for x in row) for row in crit["score"]]
        return crit.get(kind) or crit["assignment"]

    def _rubric_items(self) -> list[dict]:
        items = list(self.f.assessments)
        if self.f.capstone and not any((a.get("type") or "").lower() == "project" for a in items):
            items.append({"name": self.L["slide_capstone"], "type": "project", "weight_pct": 0})
        return items

    def s_rubrics(self, s):
        L = self.L
        out = []
        for i, a in enumerate(self._rubric_items(), start=1):
            kind = (a.get("type") or "assignment").lower()
            crit = self._criteria(kind)
            weight = float(a.get("weight_pct") or 0)
            per = round_half(weight / len(crit)) if weight else 0
            rows = []
            for k, (name, *desc) in enumerate(crit):
                pts = per if k < len(crit) - 1 else round(weight - per * (len(crit) - 1), 1)
                rows.append([f"{name} ({pts:g})" if weight else name, *desc])
            los = [self.outcome_label(j) for j, o in enumerate(self.f.outcomes, 1)
                   if a["name"] in outcome_assessments(o, self.f.assessments)]
            if kind == "project" and not los:
                los = [self.outcome_label(j) for j in range(1, len(self.f.outcomes) + 1)]
            out.append(self.sec(s, key=f"{s.key}:{i}", heading=s.title(self.lang, title=a["name"]),
                                paragraphs=[self.t("rubric_intro", w=weight, pts=weight, los=", ".join(los) or "-")]
                                if weight else [],
                                table=self.table(L["rubric_cols"], rows, [1.6, 1.5, 1.5, 1.5, 1.5])))
        return out

    # ------------------------------------------------------------------ capstone brief
    def s_capstone_brief__overview(self, s):
        return self.s_capstone(s)

    def s_scenario(self, s):
        cap = self.f.capstone or self.t("capstone_default", title=self.f.title, ind=self.ind)
        need = cap.split(". ")[0].rstrip(".") + "."
        return self.sec(s, paragraphs=[self.t("scenario", ind=self.ind, country=self.country, need=need)])

    def s_requirements(self, s):
        bullets = []
        for n, m in self.f.teaching_modules:
            tops = self.topics(n, m)
            detail = (": " if self.lang == "en" else ": ") + tops[0] if tops else ""
            bullets.append(self.t("req_module", m=m["title"], detail=detail))
        return self.sec(s, bullets=bullets + list(self.L["req_general"]))

    def s_deliverables(self, s):
        return self.sec(s, bullets=[b.format(artifact=self.artifact) for b in self.L["deliverables"]], numbered=True)

    def s_milestones(self, s):
        L = self.L
        cap_mods = [m for n, m in enumerate(self.f.modules, 1) if (n, m) not in self.f.teaching_modules]
        cap_h = sum(float(m.get("hours") or 0) for m in cap_mods) or round_half(self.f.hours * 0.15)
        weeks = self.f.weeks or 1
        span = max(1, math.ceil(cap_h / HOURS_PER_WEEK))
        start = max(1, weeks - span + 1)
        shares = (0.15, 0.55, 0.15, 0.15)
        rows = []
        for k, ((name, output), share) in enumerate(zip(L["ms"], shares)):
            week = min(weeks, start + int(k * span / 4)) if k < 3 else weeks
            rows.append([name, week, f"{round_half(cap_h * share):g}", output.format(artifact=self.artifact)])
        return self.sec(s, table=self.table(L["ms_cols"], rows, [1.2, 1, 0.8, 3]))

    def s_capstone_brief__assessment(self, s):
        L = self.L
        cap = next((a for a in self.f.assessments if (a.get("type") or "").lower() == "project"), None)
        weight = float(cap["weight_pct"]) if cap else 0
        crit = L["crit"]["project"]
        per = round_half(weight / len(crit)) if weight else 0
        rows = [[name, desc[0], f"{(per if k < len(crit) - 1 else round(weight - per * (len(crit) - 1), 1)):g}"
                 if weight else "-"] for k, (name, *desc) in enumerate(crit)]
        cols = [L["rubric_cols"][0].split(" (")[0], L["rubric_cols"][1], L["weight"]]
        paras = [self.t("rubric_intro", w=weight, pts=weight, los=self.cap_los())] if weight else []
        return self.sec(s, paragraphs=paras, table=self.table(cols, rows, [2, 3.5, 0.8]))

    def cap_los(self) -> str:
        return ", ".join(self.outcome_label(i) for i, o in enumerate(self.f.outcomes, 1)
                         if outcome_domain(o) & {"practical", "intellectual", "transferable"}) or "-"

    def s_submission(self, s):
        return self.sec(s, bullets=list(self.L["submission"]))

    def s_integrity(self, s):
        return self.sec(s, bullets=list(self.L["integrity"]))

    # ------------------------------------------------------------------ outcomes & assessment plan
    def s_intro(self, s):
        return self.sec(s, paragraphs=[self.t("plan_intro", title=self.f.title)])

    def s_outcome_map(self, s):
        L = self.L
        mods = self.f.teaching_modules
        rows = []
        for i, o in enumerate(self.f.outcomes, 1):
            taught = outcome_modules(o, mods)
            assessed = outcome_assessments(o, self.f.assessments)
            rows.append([self.outcome_label(i), o, L["bloom"][bloom_level(o)],
                         ", ".join(f"{L['module']} {n}" for n in taught) or L["cov_all_modules"],
                         "; ".join(assessed) or "-"])
        return self.sec(s, table=self.table(L["map_cols"], rows, [0.5, 3, 0.9, 1.4, 2]))

    def s_coverage(self, s):
        L = self.L
        bullets = []
        for i, o in enumerate(self.f.outcomes, 1):
            if not outcome_assessments(o, self.f.assessments):
                bullets.append(self.t("cov_missing", n=i, o=o))
            if not outcome_domain(o):
                bullets.append(self.t("cov_verb", n=i))
        if not bullets:
            bullets.append(L["cov_ok"])
        total = sum(float(a["weight_pct"]) for a in self.f.assessments)
        bullets.append(L["cov_weights_ok"] if abs(total - 100) < 0.05 else self.t("cov_weights", w=total))
        return self.sec(s, bullets=bullets)


def build_sections(facts: CourseFacts, template: tpl.Template, lang: str, client: Optional[str] = None) -> list[DocSection]:
    b = Builder(facts, template, lang)
    b._client = client
    return b.build()
