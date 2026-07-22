import streamlit as st
import pickle
import joblib
import re
import fitz
import docx

from PIL import Image
import pytesseract

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# LOAD TF-IDF
# ==========================================================

tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ==========================================================
# HR JOB OPENINGS
# ==========================================================

job_openings = {

    "Data Scientist": """
    Python Machine Learning Deep Learning TensorFlow
    PyTorch Statistics SQL NLP Computer Vision
    Data Analysis Pandas NumPy
    """,

    "Machine Learning Engineer": """
    Python MLOps Docker Kubernetes AWS
    TensorFlow PyTorch Model Deployment
    CI CD MLflow
    """,

    "Data Analyst": """
    SQL Excel Tableau Power BI
    Data Visualization Dashboard Reporting
    ETL Business Intelligence
    """,

    "Software Engineer": """
    Python Java C++
    JavaScript React NodeJS
    Data Structures Algorithms
    OOP REST API Git
    """
}

# ==========================================================
# STOPWORDS
# ==========================================================

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of",
    "with","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may",
    "might","shall","can","need","it","its","this","that"
}

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)

# ==========================================================
# CREATE JOB VECTORS
# ==========================================================

job_titles = list(job_openings.keys())

job_descriptions = [
    clean_text(desc)
    for desc in job_openings.values()
]

job_vectors = tfidf.transform(
    job_descriptions
)

# ==========================================================
# FILE EXTRACTION
# ==========================================================

def extract_pdf(file):

    doc = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text


def extract_docx(file):

    doc = docx.Document(file)

    text = "\n".join(
        [p.text for p in doc.paragraphs]
    )

    return text


def extract_image(file):

    image = Image.open(file)

    text = pytesseract.image_to_string(
        image
    )

    return text


def read_resume(file):

    name = file.name.lower()

    if name.endswith(".pdf"):
        return extract_pdf(file)

    elif name.endswith(".docx"):
        return extract_docx(file)

    elif name.endswith(
        (".png", ".jpg", ".jpeg")
    ):
        return extract_image(file)

    return ""

# ==========================================================
# HR MATCHING FUNCTION
# ==========================================================

def match_jobs(resume_text):

    cleaned_resume = clean_text(
        resume_text
    )

    resume_vector = tfidf.transform(
        [cleaned_resume]
    )

    similarities = cosine_similarity(
        resume_vector,
        job_vectors
    )[0]

    results = []

    for i, score in enumerate(
        similarities
    ):

        results.append(
            (
                job_titles[i],
                round(score * 100, 2)
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(
    page_title="HR Recruitment System",
    page_icon="💼"
)

st.title(
    "💼 AI-Powered HR Recruitment System"
)

st.write(
    "Upload Resume and Check Job Matching"
)

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=[
        "pdf",
        "docx",
        "png",
        "jpg",
        "jpeg"
    ]
)

# ==========================================================
# MAIN LOGIC
# ==========================================================

if uploaded_file:

    resume_text = read_resume(
        uploaded_file
    )

    if not resume_text.strip():

        st.error(
            "Could not extract text from file"
        )

    else:

        results = match_jobs(
            resume_text
        )

        st.subheader(
            "📊 HR Matching Report"
        )

        for role, match in results:

            st.write(
                f"**{role}** → {match}%"
            )

            st.progress(
                min(int(match), 100)
            )

            if match >= 60:

                st.success(
                    f"✅ Recommended for {role}"
                )

            else:

                st.warning(
                    f"❌ Below 60% Threshold"
                )

        # ===================================
        # BEST MATCH
        # ===================================

        best_role = results[0][0]
        best_score = results[0][1]

        st.markdown("---")

        st.subheader(
            "🎯 Final HR Decision"
        )

        if best_score >= 60:

            st.success(
                f"""
                Candidate is Recommended

                Job Role: {best_role}

                Match Score: {best_score}%
                """
            )

        else:

            st.error(
                f"""
                Candidate Not Recommended

                Highest Match:
                {best_role}

                Score:
                {best_score}%
                """
            )