import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# SKILL VOCABULARY
# ================================
SKILL_VOCAB = [
    "python", "sql", "excel", "power bi", "tableau",
    "data analysis", "data visualization", "statistics",
    "etl", "machine learning", "deep learning",
    "data engineering", "data pipeline", "big data",
    "hadoop", "spark", "gis", "mapping", "cartography",
    "automation", "algorithms", "routing",
    "database management", "data validation",
    "kpi tracking", "business intelligence",
    "communication", "reporting"
]

# ================================
# CLEAN TEXT
# ================================
def clean_text(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================================
# SKILL EXTRACTION
# ================================
skill_vectorizer = TfidfVectorizer().fit(SKILL_VOCAB)
skill_matrix = skill_vectorizer.transform(SKILL_VOCAB)

def extract_skills_light(text, threshold=0.3):
    if not isinstance(text, str) or not text.strip():
        return set()

    text = clean_text(text)
    text_vec = skill_vectorizer.transform([text])
    similarities = cosine_similarity(text_vec, skill_matrix)[0]

    return {
        SKILL_VOCAB[i]
        for i, score in enumerate(similarities)
        if score >= threshold
    }

# ================================
# BUILD JOB MATRICES
# ================================
def build_job_pipeline(jobs_df):

    jobs_df["Job Title"] = jobs_df["Job Title"].fillna("")
    jobs_df["Description"] = jobs_df["Description"].fillna("")

    jobs_df["clean_title"] = jobs_df["Job Title"].apply(clean_text)
    jobs_df["clean_description"] = jobs_df["Description"].apply(clean_text)

    jobs_df["combined_text"] = (
        jobs_df["clean_title"] + " " + jobs_df["clean_description"]
    )

    jobs_df["skills"] = jobs_df["combined_text"].apply(extract_skills_light)

    desc_vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    title_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    desc_matrix = desc_vectorizer.fit_transform(jobs_df["clean_description"])
    title_matrix = title_vectorizer.fit_transform(jobs_df["clean_title"])

    return jobs_df, desc_vectorizer, title_vectorizer, desc_matrix, title_matrix