# Resume–Job Description Analyzer using RAG & GenAI

A **Resume and Job Description (JD) Analyzer** built using **Retrieval-Augmented Generation (RAG)** and **Generative AI**.  
This application evaluates how well a resume matches a job description by computing an **ATS score**, identifying **matched and missing skills**, suggesting **resume improvements**, and recommending the **best job role**.

---

## 🚀 Features

- 📊 **ATS Matching Score (0–100)**
- ✅ **Matched Skills**
- ❌ **Missing Skills**
- 💡 **Resume Suggestions & Improvements**
- 🏆 **Best Job Role Recommendation**
- 📄 **PDF Resume Upload**
- 🧠 **RAG-based contextual reasoning**
- 🌐 **Streamlit UI**

---

## 🧠 Architecture

![Resume–JD Analyzer Architecture](screenshots/arc.png)

This system follows a **Retrieval-Augmented Generation (RAG)** pipeline:

- The **resume (PDF)** and **job description (text)** are converted into embeddings.
- **Cosine similarity** between embeddings is used to calculate the **ATS score (0–100)**.
- A **FAISS vector database** stores job descriptions and retrieves relevant context.
- **LLaMA-3** uses the resume, JD, and retrieved context to generate:
  - Matched skills
  - Missing skills
  - Resume improvement suggestions
  - Best-matching job role


---

## 🛠️ Tech Stack

- LLaMA-3 (Ollama)
- LangChain
- FAISS
- HuggingFace Embeddings
- Streamlit
- PDFPlumber
- NumPy

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
python ingest_jobs.py
streamlit run app.py
```

---

## 👤 Author

Vyshnav S
