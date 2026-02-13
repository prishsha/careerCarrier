import re
from collections import defaultdict

import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
MIN_CONFIDENCE_SCORE = 0.18

# loading the data
jobs_df = pd.read_csv("data/data.csv")
courses_df = pd.read_csv("data/Coursera.csv")

# normalized skill dictionary with aliases for better recall
SKILL_ALIASES = {
    "python": ["python", "pandas", "numpy", "scikit learn", "sklearn"],
    "sql": ["sql", "postgresql", "mysql", "sqlite", "snowflake", "bigquery"],
    "excel": ["excel", "spreadsheet", "google sheets", "microsoft excel"],
    "power bi": ["power bi", "powerbi", "dax"],
    "tableau": ["tableau"],
    "communication": ["communication", "presentation", "stakeholder", "storytelling"],
    "reporting": ["reporting", "report development", "report writing"],
    "dashboard": ["dashboard", "dashboards", "kpi tracking"],
    "data analysis": ["data analysis", "analysis", "analytical", "analytics"],
    "data visualization": ["data visualization", "visualization", "data viz"],
    "statistics": ["statistics", "statistical", "hypothesis testing", "ab testing"],
    "etl": ["etl", "data pipeline", "data cleaning", "data wrangling"],
}


def clean_text(text):
    """Normalize free text for matching."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# clean coursera dataset
def clean_course_skills(skill_string):
    if pd.isna(skill_string):
        return []

    skill_string = skill_string.strip("{}")
    skills = skill_string.split(",")
    cleaned = [clean_text(s.replace('"', "").strip()) for s in skills]
    return cleaned


def extract_skills(text):
    """Extract canonical skills using alias phrase matching."""
    normalized = clean_text(text)
    found = set()
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                found.add(canonical)
                break
    return found


courses_df["skills_list"] = courses_df["skills"].apply(clean_course_skills)
courses_df["rating"] = pd.to_numeric(courses_df["rating"], errors="coerce").fillna(0.0)
courses_df["reviewcount_num"] = (
    courses_df["reviewcount"]
    .astype(str)
    .str.lower()
    .str.replace(",", "", regex=False)
    .str.replace("k", "000", regex=False)
    .str.extract(r"(\d+)", expand=False)
    .fillna("0")
    .astype(int)
)

jobs_df["Job Title"] = jobs_df["Job Title"].fillna("")
jobs_df["Description"] = jobs_df["Description"].fillna("")
jobs_df["clean_title"] = jobs_df["Job Title"].apply(clean_text)
jobs_df["clean_description"] = jobs_df["Description"].apply(clean_text)
jobs_df["skills"] = jobs_df["clean_description"].apply(extract_skills)

# separate vectorizers keep job titles influential for intent alignment
desc_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
title_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
desc_matrix = desc_vectorizer.fit_transform(jobs_df["clean_description"])
title_matrix = title_vectorizer.fit_transform(jobs_df["clean_title"])


def rank_jobs(resume_text, top_n=5):
    clean_resume = clean_text(resume_text)
    resume_skills = extract_skills(clean_resume)

    resume_desc_vector = desc_vectorizer.transform([clean_resume])
    resume_title_vector = title_vectorizer.transform([clean_resume])

    desc_scores = cosine_similarity(resume_desc_vector, desc_matrix)[0]
    title_scores = cosine_similarity(resume_title_vector, title_matrix)[0]

    # blend text similarity + explicit skill overlap
    scored = []
    for idx, row in jobs_df.iterrows():
        job_skills = row["skills"]
        if job_skills:
            overlap = len(resume_skills.intersection(job_skills)) / len(job_skills)
        else:
            overlap = 0.0

        final_score = (0.55 * desc_scores[idx]) + (0.20 * title_scores[idx]) + (0.25 * overlap)
        scored.append((idx, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n], resume_skills


def normalize_course_skills(skills_list):
    mapped = set()
    joined = " ".join(skills_list)
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", joined):
                mapped.add(canonical)
                break
    return mapped


courses_df["normalized_skills"] = courses_df["skills_list"].apply(normalize_course_skills)

def recommend_courses(missing_skills, courses_df, top_n=2):
    recommendations = defaultdict(list)

    for skill in missing_skills:
        matched_df = courses_df[courses_df["normalized_skills"].apply(lambda x: skill in x)].copy()
        if matched_df.empty:
            recommendations[skill] = []
            continue

        matched_df = matched_df.sort_values(by=["rating", "reviewcount_num"], ascending=[False, False])
        recommendations[skill] = matched_df[["course", "rating", "level"]].head(top_n).to_dict(orient="records")

    return recommendations


# route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        resume_text = request.form["resume"]
        ranked_jobs, resume_skills = rank_jobs(resume_text, top_n=5)
        top_index, top_score = ranked_jobs[0]
        best_job = jobs_df.iloc[top_index]["Job Title"]
        best_job_skills = jobs_df.iloc[top_index]["skills"]

        matched = resume_skills.intersection(best_job_skills)
        missing = best_job_skills.difference(resume_skills)

        course_recommendations = recommend_courses(missing, courses_df)
        top_matches = [
            {
                "job": jobs_df.iloc[idx]["Job Title"],
                "score": round(score * 100, 1),
            }
            for idx, score in ranked_jobs
        ]
        note = None
        if top_score < MIN_CONFIDENCE_SCORE:
            note = (
                "Low-confidence match. Add more role-specific skills/tools in the resume text "
                "to improve recommendation quality."
            )

        result = {
            "job": best_job,
            "score": round(top_score * 100, 1),
            "matched": sorted(matched),
            "missing": sorted(missing),
            "courses": dict(course_recommendations),
            "top_matches": top_matches,
            "note": note,
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
