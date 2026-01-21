import streamlit as st
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from numpy import dot
from numpy.linalg import norm

# UI Setup 
st.set_page_config(page_title="Resume + JD Analyzer",layout="wide")
st.title("📄 Resume + Job Description Analyzer")

# accpets Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

#  accepts Job Description 
jd_input = st.text_area(
    "Paste Job Description Here",
    height=300,
    placeholder="Paste the full job description here..."
)


# Extract Resume Text 
def extract_text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

resume_text = None
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

# Load FAISS  and Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# ---------- Load LLaMA-3 ----------
llm = Ollama(model="llama3", temperature=0.3)

# ---------- Run Analysis ----------
if jd_input and resume_text:
    with st.spinner("Analyzing Resume & Job Description..."):

        # ---------- Step 1: Compute ATS Score ----------
        jd_emb = embedding_model.embed_query(jd_input)
        resume_emb = embedding_model.embed_query(resume_text)

        cos_sim = dot(jd_emb, resume_emb) / (norm(jd_emb) * norm(resume_emb))
        ats_score = round(cos_sim * 100, 2)
        st.subheader("📊 ATS Matching Score")
        st.write(f"{ats_score} / 100")

        # ---------- Step 2: Retrieve top-K similar job postings from FAISS ----------
        docs = vectorstore.similarity_search(jd_input, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # ---------- Step 3: Prepare prompt for LLaMA-3 ----------
        prompt = f"""
You are an Strict AI job matching assistant.

Job Description:
{jd_input}

Candidate Resume:
{resume_text}

Context from similar job postings:
{context}

please provide the output in the following format:

1. Matched Skills: bullet points
2. Missing Skills: bullet points
3. Suggestions and Improvements: paragraph(s)
4. Top Best Jobs: bullet points (top 3 job titles only)
"""

        # ---------- Step 4: Invoke LLaMA-3 ----------
        response = llm.invoke(prompt)

        # ---------- Step 5: Parse LLaMA response ----------
        sections = {"Matched Skills": [], "Missing Skills": [], 
                    "Suggestions and Improvements": "", "Top Best Jobs": []}

        current_section = None
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if "Matched Skills" in line:
                current_section = "Matched Skills"
                continue
            elif "Missing Skills" in line:
                current_section = "Missing Skills"
                continue
            elif "Suggestions and Improvements" in line:
                current_section = "Suggestions and Improvements"
                continue
            elif "Top Best Jobs" in line:
                current_section = "Top Best Jobs"
                continue

            if current_section in ["Matched Skills", "Missing Skills", "Top Best Jobs"]:
                if line.startswith("-"):
                    sections[current_section].append(line)
                else:
                    sections[current_section].append(f"- {line}")
            elif current_section == "Suggestions and Improvements":
                sections[current_section] += line + " "

        # ---------- Step 6: Display Sections ----------
        st.subheader("✅ Matched Skills")
        if sections["Matched Skills"]:
            for skill in sections["Matched Skills"]:
                st.write(skill)
        else:
            st.write("- N/A")

        st.subheader("❌ Missing Skills")
        if sections["Missing Skills"]:
            for skill in sections["Missing Skills"]:
                st.write(skill)
        else:
            st.write("- N/A")

        st.subheader("💡 Suggestions and Improvements")
        st.write(sections["Suggestions and Improvements"].strip() or "N/A")

        st.subheader("🏆 Top Best Jobs")
        if sections["Top Best Jobs"]:
            for job in sections["Top Best Jobs"]:
        # Keep only the job title (first part before any 'at' or '-')
                title = job.replace("-", "").strip()
        # If it contains 'at', take only the part before 'at'
                if " at " in title:
                    title = title.split(" at ")[0].strip()
                st.write(f"- {title}")
        else:
             st.write("- N/A")

