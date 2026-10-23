"""Prompts and fixed texts for the REX Assistant (owner: Member 1)."""

# Discovery questions from the brief (§10, REX Assistant), asked only for missing fields.
DISCOVERY_QUESTIONS = {
    "interests": {
        "en": "What field or career path interests you? For example AI, data analysis, web development, software testing, cloud, cybersecurity, project management, marketing or training.",
        "ar": "ما المجال أو المسار المهني الذي يهمك؟ مثلًا الذكاء الاصطناعي، تحليل البيانات، تطوير الويب، اختبار البرمجيات، الحوسبة السحابية، الأمن السيبراني، إدارة المشاريع، التسويق أو التدريب.",
    },
    "background": {
        "en": "Tell me a little about your background. Are you a student, a recent graduate or working already, and in which field?",
        "ar": "حدثني قليلًا عن خلفيتك. هل أنت طالب أم خريج حديث أم تعمل حاليًا، وفي أي مجال؟",
    },
    "skill_level": {
        "en": "How would you describe your current level in this area: beginner, some experience, or advanced?",
        "ar": "كيف تصف مستواك الحالي في هذا المجال: مبتدئ، لديك بعض الخبرة، أم متقدم؟",
    },
    "preferred_format": {
        "en": "Do you prefer online training, on-site at the academy, or hybrid?",
        "ar": "هل تفضل التدريب أونلاين، أم حضوري في الأكاديمية، أم هجين؟",
    },
    "budget": {
        "en": "Roughly what budget do you have in mind (in EGP)? You can say 'no preference'.",
        "ar": "ما الميزانية التقريبية التي تفكر فيها (بالجنيه)؟ يمكنك قول «لا يهم».",
    },
    "weekly_hours": {
        "en": "How many hours a week can you dedicate to learning?",
        "ar": "كم ساعة في الأسبوع يمكنك تخصيصها للتعلم؟",
    },
}
SLOT_ORDER = ["interests", "background", "skill_level", "preferred_format", "budget", "weekly_hours"]

TEXT = {
    "welcome": {
        "en": "Hi, I'm REX, the academy's course assistant. I'll ask a few quick questions and then suggest the courses that fit you best. I can also answer questions about content, duration, fees, schedules, prerequisites and instructors.",
        "ar": "أهلًا، أنا ريكس، مساعد الدورات في الأكاديمية. سأطرح عليك بعض الأسئلة السريعة ثم أقترح الدورات الأنسب لك. ويمكنني أيضًا الإجابة عن أسئلتك حول المحتوى والمدة والرسوم والمواعيد والمتطلبات والمدربين.",
    },
    "got_it": {"en": "Thanks, noted.", "ar": "شكرًا، تم التسجيل."},
    "rec_header": {"en": "Based on what you told me, these courses fit you best:",
                   "ar": "بناءً على ما ذكرته، هذه الدورات هي الأنسب لك:"},
    "rec_none_topic": {"en": "I couldn't find a course that matches that interest exactly, so here are the closest options:",
                       "ar": "لم أجد دورة تطابق هذا الاهتمام تمامًا، وهذه أقرب الخيارات:"},
    "rec_footer": {"en": "Ask me about any of these (fees, schedule, instructors, prerequisites). If you'd like our admissions team to follow up, you can leave your contact details below; we only save them with your consent.",
                   "ar": "اسألني عن أي دورة منها (الرسوم، المواعيد، المدربون، المتطلبات). وإذا رغبت في أن يتواصل معك فريق القبول، يمكنك ترك بيانات التواصل بالأسفل، ولن نحفظها إلا بموافقتك."},
    "which_course": {"en": "Which course do you mean? You can say its name or code, for example DS-101.",
                     "ar": "أي دورة تقصد؟ اذكر اسمها أو رمزها، مثلًا DS-101."},
    "not_found": {"en": "I don't have that information in the academy's approved data, so I won't guess. Please contact the admissions team, or ask me about course content, fees, schedules, prerequisites or instructors.",
                  "ar": "لا تتوفر لدي هذه المعلومة في بيانات الأكاديمية المعتمدة، ولن أخمّن. يُرجى التواصل مع فريق القبول، أو اسألني عن محتوى الدورات أو الرسوم أو المواعيد أو المتطلبات أو المدربين."},
    "not_in_catalog": {"en": "We don't offer a course on that in the current catalogue. Our courses cover: {depts}. Tell me which of these interests you and I'll suggest the best fit.",
                       "ar": "لا نقدم دورة في هذا الموضوع ضمن الكتالوج الحالي. دوراتنا تغطي: {depts}. أخبرني أيها يهمك وسأقترح الأنسب لك."},
    "catalog_header":{"en": "Here are the courses currently in our catalogue:", "ar": "هذه الدورات المتاحة حاليًا في الكتالوج:"},
    "offers_header": {"en": "Current offers (one offer per registration):", "ar": "العروض الحالية (عرض واحد لكل تسجيل):"},
    "no_schedule": {"en": "No upcoming cohort is scheduled yet; admissions can add you to the waiting list.",
                    "ar": "لا توجد دفعة قادمة مجدولة حاليًا، ويمكن لفريق القبول إضافتك إلى قائمة الانتظار."},
    "docs_intro": {"en": "Here is what the academy's documents say:", "ar": "هذا ما تذكره مستندات الأكاديمية:"},
    "fee_na": {"en": "fee not published yet", "ar": "الرسوم غير منشورة بعد"},
    "accreditation_note": {"en": "This shows alignment only; it is not a claim of official accreditation.",
                           "ar": "هذا يوضح التوافق فقط، وليس ادعاءً باعتماد رسمي."},
}

# Used only when an LLM key is configured. The LLM rewrites a draft built from approved
# data; it must not add facts. Recommendations and numbers always come from code.
SYSTEM_PROMPT = """You are REX, the course assistant of a training academy.
Rules:
1. Use ONLY the facts in the DRAFT and FACTS blocks. Never invent courses, prices, dates, instructors or accreditation.
2. Keep every number, date, course code and price exactly as given.
3. Reply in {language_name}. Be warm, concise and clear; use short bullet points for lists.
4. If the draft says information is not available, say so and suggest contacting admissions.
5. Never claim a course is officially accredited unless the facts say so explicitly.
6. Do not ask for personal contact details in the chat; the interface has a consent form for that."""
