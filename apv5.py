import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="شات بوت الصحة النفسية", layout="centered", page_icon="🌿")

st.title("🌿 شات بوت تقييم الصحة النفسية")
st.markdown("### استبيانات طبية موثوقة للكشف المبكر عن اضطرابات نفسية شائعة")
st.info("⚠️ هذه الأداة لأغراض تعليمية وبحثية فقط، ولا تغني عن استشارة طبيب نفسي أو أخصائي.")

MODELS_DIR = "models"

assessments = {
    "1": {
        "name": "القلق العام (GAD-7)",
        "prefix": "GAD-7",
        "max_answer": 3,
        "num_questions": 7,
        "answer_type": "frequency",
        "questions": [
            "حسيت بالتوتر أو القلق أو العصبية الزايدة في آخر أسبوعين؟",
            "مقدرتش أوقف أو أسيطر على القلق والتفكير الزايد؟",
            "قلقت زيادة عن اللزوم على حاجات كتير مختلفة؟",
            "واجهت صعوبة في الاسترخاء؟",
            "كنت مضطرب جدًا لدرجة إني مش قادر أقعد ساكت؟",
            "بقيت أتعصب أو أتنرفز بسرعة؟",
            "حسيت بالخوف زي كأن حاجة وحشة هتحصل؟"
        ]
    },
    "2": {
        "name": "الاكتئاب (PHQ-9)",
        "prefix": "PHQ-9",
        "max_answer": 3,
        "num_questions": 9,
        "answer_type": "frequency",
        "questions": [
            "قلة الاهتمام أو المتعة في عمل الأشياء؟",
            "الشعور بالإحباط أو اليأس أو الاكتئاب؟",
            "صعوبة في النوم، الاستغراق في النوم، أو النوم كثيرًا؟",
            "الشعور بالتعب أو قلة الطاقة؟",
            "قلة الشهية أو الأكل كثيرًا؟",
            "الشعور بالسوء عن نفسك أو أنك فاشل؟",
            "صعوبة في التركيز (مثل القراءة أو مشاهدة التلفزيون)؟",
            "الحركة ببطء أو العكس (السرعة الزايدة والقلق الحركي)؟",
            "أفكار بأنك أفضل تموت أو إيذاء نفسك بطريقة ما؟"
        ]
    },
    "3": {
        "name": "التوتر المدرك (PSS-10)",
        "prefix": "PSS-10",
        "max_answer": 4,
        "num_questions": 10,
        "answer_type": "frequency5",
        "questions": [
            "في الشهر الأخير، كم مرة شعرت بالانزعاج بسبب حدث غير متوقع؟",
            "كم مرة شعرت أنك غير قادر على السيطرة على الأمور المهمة في حياتك؟",
            "كم مرة شعرت بالتوتر والعصبية؟",
            "كم مرة تعاملت بنجاح مع المضايقات اليومية؟",
            "كم مرة شعرت أنك تتعامل بفعالية مع التغييرات المهمة في حياتك؟",
            "كم مرة شعرت بالثقة في قدرتك على التعامل مع مشاكلك الشخصية؟",
            "كم مرة شعرت أن الأمور تسير كما تريد؟",
            "كم مرة شعرت أنك غير قادر على التعامل مع كل ما يجب عليك فعله؟",
            "كم مرة شعرت أنك تسيطر على الطريقة التي تقضي بها وقتك؟",
            "كم مرة شعرت أن الصعوبات تتراكم لدرجة أنك لا تستطيع التغلب عليها؟"
        ]
    },
    "4": {
        "name": "الوسواس القهري (Y-BOCS)",
        "prefix": "Y-BOCS",
        "max_answer": 4,
        "num_questions": 10,
        "answer_type": "frequency5",
        "questions": [
            "كم من الوقت تقضيه يوميًا في الوساوس؟",
            "كم الوساوس بتعيق حياتك اليومية؟",
            "كم الضيق اللي بتسببه الوساوس؟",
            "كم بتحاول تقاوم الوساوس؟",
            "كم قدرتك على السيطرة على الوساوس؟",
            "كم من الوقت تقضيه في السلوكيات القهرية؟",
            "كم السلوكيات القهرية بتعيق حياتك؟",
            "كم الضيق اللي بتسببه السلوكيات القهرية؟",
            "كم بتحاول تقاوم السلوكيات القهرية؟",
            "كم قدرتك على السيطرة على السلوكيات القهرية؟"
        ]
    },
    "5": {
        "name": "الاضطراب ثنائي القطب (MDQ)",
        "prefix": "MDQ",
        "max_answer": 1,
        "num_questions": 13,
        "answer_type": "yesno",
        "questions": [
            "هل كنت في فترة شعرت فيها بفرح أو نشاط زايد أكثر من المعتاد؟",
            "هل كنت أكثر عصبية أو انفعالاً من المعتاد؟",
            "هل كنت أكثر ثقة في نفسك من المعتاد؟",
            "هل احتجت نوم أقل من المعتاد وما زلت نشيط؟",
            "هل كنت أكثر كلاماً من المعتاد؟",
            "هل كانت أفكارك تتسارع بسرعة كبيرة؟",
            "هل كنت مشتت التركيز بسهولة؟",
            "هل زادت أنشطتك الاجتماعية أو الجنسية بشكل ملحوظ؟",
            "هل قمت بأفعال متهورة (إنفاق زائد، قيادة متهورة، استثمارات محفوفة بالمخاطر)؟",
            "هل كانت هذه التغييرات واضحة للآخرين؟",
            "هل سببت لك هذه التغييرات مشاكل؟",
            "هل حدثت عدة تغييرات من دي في نفس الوقت؟",
            "هل كانت هذه الفترة شديدة لدرجة أنها سببت مشاكل في العمل أو العلاقات؟"
        ]
    }
}

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.answers = []
    st.session_state.current_question = 0
    st.session_state.selected = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.selected is None:
    with st.chat_message("assistant"):
        st.markdown("""
مرحبًا! 👋  
أنا شات بوت لتقييم الصحة النفسية باستخدام استبيانات طبية موثوقة.

اختر الاستبيان اللي عايز تعمله:

1. القلق العام (GAD-7)  
2. الاكتئاب (PHQ-9)  
3. التوتر المدرك (PSS-10)  
4. الوسواس القهري (Y-BOCS)  
5. الاضطراب ثنائي القطب (MDQ)

**اكتب الرقم فقط**
        """)

if prompt := st.chat_input("اكتب رسالتك هنا..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.selected is None:
        if prompt in assessments:
            st.session_state.selected = prompt
            ass = assessments[prompt]
            st.session_state.current_question = 1

            if ass["answer_type"] == "yesno":
                options = "0 → لا\n1 → نعم"
            elif ass["answer_type"] == "frequency5":
                options = "0 → لا أبدًا\n1 → نادرًا\n2 → أحيانًا\n3 → غالبًا\n4 → دائمًا"
            else:
                options = "0 → خالص مش حصل\n1 → أيام قليلة\n2 → أكتر من نص الأيام\n3 → تقريبًا كل يوم"

            with st.chat_message("assistant"):
                st.markdown(f"تمام! اخترت **{ass['name']}**\n\n**سؤال 1 من {ass['num_questions']}:**\n{ass['questions'][0]}\n\n{options}")
        else:
            with st.chat_message("assistant"):
                st.markdown("لو سمحت اختر رقم من 1 إلى 5 😊")

    else:
        ass = assessments[st.session_state.selected]
        max_ans = ass["max_answer"]

        if prompt.isdigit() and 0 <= int(prompt) <= max_ans:
            st.session_state.answers.append(int(prompt))
            with st.chat_message("assistant"):
                st.markdown("تم التسجيل ✅")

            if st.session_state.current_question < ass["num_questions"]:
                q_num = st.session_state.current_question
                if ass["answer_type"] == "yesno":
                    options = "0 → لا\n1 → نعم"
                elif ass["answer_type"] == "frequency5":
                    options = "0 → لا أبدًا\n1 → نادرًا\n2 → أحيانًا\n3 → غالبًا\n4 → دائمًا"
                else:
                    options = "0 → خالص مش حصل\n1 → أيام قليلة\n2 → أكتر من نص الأيام\n3 → تقريبًا كل يوم"

                with st.chat_message("assistant"):
                    st.markdown(f"**سؤال {q_num + 1} من {ass['num_questions']}:**\n{ass['questions'][q_num]}\n\n{options}")
                st.session_state.current_question += 1

            else:
                answers = st.session_state.answers.copy()
                if "PSS-10" in ass["name"]:
                    for i in [3, 4, 5, 6, 8]:
                        if i < len(answers):
                            answers[i] = 4 - answers[i]

                total_score  = sum(answers)
                max_possible = ass["num_questions"] * ass["max_answer"]

                df = pd.DataFrame([answers], columns=[f"Q{i+1}" for i in range(len(answers))])

                model_file   = os.path.join(MODELS_DIR, f"{ass['prefix']}_model.pkl")
                encoder_file = os.path.join(MODELS_DIR, f"{ass['prefix']}_encoder.pkl")
                map_file     = os.path.join(MODELS_DIR, f"{ass['prefix']}_map.json")

                model_level_num  = "غير متاح"
                model_level_text = ""
                if all(os.path.exists(f) for f in [model_file, encoder_file, map_file]):
                    try:
                        clf        = joblib.load(model_file)
                        le         = joblib.load(encoder_file)
                        with open(map_file, "r", encoding="utf-8") as f:
                            label_map = json.load(f)
                        pred_enc   = clf.predict(df)[0]
                        pred_label = le.inverse_transform([pred_enc])[0]
                        model_level_num  = str(pred_label)
                        model_level_text = f" - {label_map.get(str(pred_enc), 'غير معروف')}"
                    except Exception as e:
                        model_level_text = f" (error: {e})"

                with st.chat_message("assistant"):
                    st.markdown(f"### خلصنا استبيان: **{ass['name']}**! 🎉")
                    st.markdown(f"**المجموع الخام:** {total_score}/{max_possible}")
                    st.markdown(f"**مستوى حسب الموديل:** {model_level_num}{model_level_text}")
                    st.markdown("### تفسير النتيجة:")

                    if "GAD-7" in ass["name"]:
                        if total_score <= 4:
                            st.success("🟢 حالتك ممتازة تمامًا ومفيش أي قلق يُذكر! استمر كده ❤️")
                        elif total_score <= 9:
                            st.info("🟡 قلق خفيف - راقب نفسك شوية")
                        elif total_score <= 14:
                            st.warning("🟠 قلق متوسط - يفضل تكلم متخصص")
                        else:
                            st.error("🔴 قلق شديد - مهم جدًا تستشير طبيب نفسي فورًا")

                    elif "PHQ-9" in ass["name"]:
                        if total_score <= 4:
                            st.success("🟢 حالتك ممتازة ومفيش اكتئاب يُذكر! استمر كده ❤️")
                        elif total_score <= 9:
                            st.info("🟡 اكتئاب خفيف")
                        elif total_score <= 14:
                            st.warning("🟠 اكتئاب متوسط")
                        elif total_score <= 19:
                            st.error("🔴 اكتئاب متوسط إلى شديد")
                        else:
                            st.error("🔴 اكتئاب شديد - مهم جدًا تستشير طبيب فورًا")

                    elif "PSS-10" in ass["name"]:
                        if total_score <= 13:
                            st.success("🟢 توتر منخفض جدًا - كويس جدًا يا بطل!")
                        elif total_score <= 26:
                            st.warning("🟠 توتر متوسط - جرب تمارين تنفس أو رياضة")
                        else:
                            st.error("🔴 توتر عالي جدًا - مهم جدًا تستشير متخصص")

                    elif "Y-BOCS" in ass["name"]:
                        if total_score <= 7:
                            st.success("🟢 لا يشير إلى وسواس قهري ملحوظ - حالتك ممتازة تمامًا! 🎉")
                        elif total_score <= 15:
                            st.info("🟡 وسواس خفيف - راقب نفسك")
                        elif total_score <= 23:
                            st.warning("🟠 وسواس متوسط - يفضل متابعة مع متخصص")
                        elif total_score <= 31:
                            st.error("🔴 وسواس شديد - مهم جدًا تستشير طبيب نفسي")
                        else:
                            st.error("🔴 وسواس شديد جدًا - مهم جدًا تستشير طبيب نفسي فورًا")

                    elif "MDQ" in ass["name"]:
                        if total_score >= 7:
                            st.error("🔴 محتمل اضطراب ثنائي القطب - مهم جدًا تستشير طبيب نفسي بأسرع وقت")
                        else:
                            st.success("🟢 غير محتمل اضطراب ثنائي القطب - كويس جدًا!")

                    st.markdown("---")
                    st.markdown("شكرًا على ثقتك وإكمال الاستبيان بصراحة ❤️\nلو عايز تعيد أو تجرب استبيان تاني، اكتب **إعادة**")

        elif prompt.lower() in ["إعادة", "ريستارت", "restart", "اعادة", "جديد", "بدء", "start"]:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        else:
            with st.chat_message("assistant"):
                st.markdown(f"لو سمحت رد برقم من 0 إلى {max_ans} فقط 😊")
