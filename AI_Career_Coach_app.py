import io
import os
import streamlit as st
import docx
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="AI Career Coach", page_icon="📈", layout="wide")

SYSTEM_PROMPT = """You are an expert career coach specializing in data careers.
Your role is to provide comprehensive, role-specific guidance that prioritizes skills based on job market demand and role specificity.

Your guidance must always include these five sections:
1. Strengths Assessment: Analyze strong areas and foundation skills
2. Resume-to-JD Fit Summary: Map existing experience to job requirements
3. Skill Gap Analysis: Identify missing or underdeveloped skills
4. Interview Preparation Guidance: Provide targeted interview tips for this role
5. Upskilling Roadmap: Suggest learning priorities and resources

When evaluating skills, always distinguish between:
- High Priority: Critical skills most demanded for this role
- Medium Priority: Important but less frequently required skills
- Low Priority: Nice-to-have or emerging skills

Always ground your advice in the specific skill priorities provided for the role."""

REQUIRED_SECTIONS = [
    "Strengths Assessment:",
    "Resume-to-JD Fit Summary:",
    "Skill Gap Analysis:",
    "Interview Preparation Guidance:",
    "Upskilling Roadmap:"
]
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages).strip()

    return ""

def parse_csv_skills(text):
    return [x.strip() for x in text.split(",") if x.strip()]

def build_user_prompt(
    role,
    resume_text,
    job_description_text,
    high_skills,
    med_skills,
    low_skills,
    evidence_text
):
    high_line = ", ".join(high_skills) if high_skills else "Not provided"
    med_line = ", ".join(med_skills) if med_skills else "Not provided"
    low_line = ", ".join(low_skills) if low_skills else "Not provided"

    prompt = f"""
Role bucket: {role}

Weighted Skill Profile:
High Priority: {high_line}
Medium Priority: {med_line}
Low Priority: {low_line}

Requirement Evidence:
{evidence_text if evidence_text.strip() else "No extra evidence provided."}

Job Description:
{job_description_text if job_description_text.strip() else "No job description provided."}

Candidate Resume / Background:
{resume_text if resume_text.strip() else "No resume text provided. Provide role-focused coaching based on skills only."}

Generate detailed coaching with all five required sections.
""".strip()
    return prompt

def build_context_prompt(
    role,
    resume_text,
    job_description_text,
    high_skills,
    med_skills,
    low_skills,
    evidence_text
):
    high_line = ", ".join(high_skills) if high_skills else "Not provided"
    med_line = ", ".join(med_skills) if med_skills else "Not provided"
    low_line = ", ".join(low_skills) if low_skills else "Not provided"

    return f"""
Current coaching context:

Role bucket: {role}

Weighted Skill Profile:
High Priority: {high_line}
Medium Priority: {med_line}
Low Priority: {low_line}

Requirement Evidence:
{evidence_text if evidence_text.strip() else "No extra evidence provided."}

Job Description:
{job_description_text if job_description_text.strip() else "No job description provided."}

Candidate Resume / Background:
{resume_text if resume_text.strip() else "No resume text provided. Provide role-focused coaching based on skills only."}

Use this context to answer the user's follow-up question conversationally. If they ask for a full review, still organize the answer into the five required sections.
""".strip()

def has_all_sections(text):
    return all(section in text for section in REQUIRED_SECTIONS)

def call_model(api_key, model_name, messages, temperature=0.3, max_tokens=900):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

st.title("AI Career Coach - Fine-Tuned Model")
st.caption("Role-specific coaching using weighted skill priorities and your fine-tuned OpenAI model.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_context_signature" not in st.session_state:
    st.session_state.chat_context_signature = ""

with st.sidebar:
    st.subheader("Configuration")
    api_key_input = st.text_input(
        "OPENAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )

    default_model = os.getenv("FINE_TUNED_MODEL", "")
    model_name = st.text_input(
        "Fine-tuned model name",
        value=default_model,
        placeholder="ft:gpt-4.1-nano-2025-04-14:org:project:career-coach..."
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    max_tokens = st.slider("Max tokens", min_value=300, max_value=1500, value=900, step=50)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Inputs")

    role = st.selectbox(
        "Select Role Bucket",
        ["Data Analyst", "Data Engineer", "BI/Analytics", "Risk/Credit"]
    )

    resume_file = st.file_uploader(
        "Upload Resume (.docx, .pdf, .txt)",
        type=["docx", "pdf", "txt"]
    )
    jd_file = st.file_uploader(
        "Upload Job Description (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"]
    )

    uploaded_resume_text = read_uploaded_file(resume_file) if resume_file else ""
    uploaded_jd_text = read_uploaded_file(jd_file) if jd_file else ""

    resume_text = st.text_area(
        "Resume Text (editable)",
        value=uploaded_resume_text,
        height=220,
        placeholder="Upload a resume or paste text..."
    )

    job_description_text = st.text_area(
        "Job Description Text (editable)",
        value=uploaded_jd_text,
        height=220,
        placeholder="Upload a JD or paste text..."
    )

    high_input = st.text_area(
        "High Priority Skills (comma-separated)",
        value="sql, python, statistics, data visualization"
    )

    med_input = st.text_area(
        "Medium Priority Skills (comma-separated)",
        value="etl, cloud, dashboarding, experimentation"
    )

    low_input = st.text_area(
        "Low Priority Skills (comma-separated)",
        value="genai, mlops, prompt engineering"
    )

    evidence_input = st.text_area(
        "Requirement Evidence (optional)",
        height=120,
        placeholder="- must have strong SQL\n- preferred cloud exposure\n- required analytics storytelling"
    )
    st.caption("Set the context here, then use the chat on the right to ask for coaching or follow-up questions.")

with col2:
    st.subheader("Chat")
    st.caption("Ask for a full review or a follow-up question. The assistant will use the context you set on the left.")

    high_skills = parse_csv_skills(high_input)
    med_skills = parse_csv_skills(med_input)
    low_skills = parse_csv_skills(low_input)

    context_signature = "||".join([
        role,
        resume_text,
        job_description_text,
        high_input,
        med_input,
        low_input,
        evidence_input,
    ])

    if st.session_state.chat_context_signature != context_signature:
        st.session_state.chat_messages = []
        st.session_state.chat_context_signature = context_signature

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.markdown("Share what you want to improve, and I’ll coach you using the role, resume, JD, and skill priorities you provided.")

    user_message = st.chat_input("Ask for resume feedback, gap analysis, interview prep, or a follow-up...")

    if user_message:
        if not api_key_input:
            st.error("Please provide OPENAI API key.")
        elif not model_name.strip():
            st.error("Please provide your fine-tuned model name.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_message})

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "system",
                    "content": build_context_prompt(
                        role=role,
                        resume_text=resume_text,
                        job_description_text=job_description_text,
                        high_skills=high_skills,
                        med_skills=med_skills,
                        low_skills=low_skills,
                        evidence_text=evidence_input,
                    ),
                },
            ]
            messages.extend(st.session_state.chat_messages)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        output = call_model(
                            api_key=api_key_input,
                            model_name=model_name.strip(),
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        st.markdown(output)
                        section_ok = has_all_sections(output)
                        st.caption("Section compliance: " + ("Pass" if section_ok else "Fail"))

                        missing = [s for s in REQUIRED_SECTIONS if s not in output]
                        if missing:
                            st.warning("Missing sections: " + ", ".join(missing))

                        st.session_state.chat_messages.append({"role": "assistant", "content": output})
                        st.rerun()

                    except Exception as e:
                        error_message = f"Error while calling model: {e}"
                        st.error(error_message)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_message})

with st.expander("Prompt Preview"):
    high_skills = parse_csv_skills(high_input)
    med_skills = parse_csv_skills(med_input)
    low_skills = parse_csv_skills(low_input)
    st.code(
        build_user_prompt(
            role=role,
            resume_text=resume_text,
            job_description_text=job_description_text,
            high_skills=high_skills,
            med_skills=med_skills,
            low_skills=low_skills,
            evidence_text=evidence_input
        )
    )