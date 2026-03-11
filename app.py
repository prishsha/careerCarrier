from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd

from preprocess import load_and_combine_courses
from data_processing import (
    build_job_pipeline,
    extract_skills_light
)

app = Flask(__name__)
MIN_CONFIDENCE_SCORE = 0.18

# ================================
# LOAD JOB DATA + BUILD PIPELINE
# ================================
jobs_df = pd.read_csv("data/data.csv")

jobs_df, desc_vectorizer, title_vectorizer, desc_matrix, title_matrix = (
    build_job_pipeline(jobs_df)
)

# ================================
# LOAD COURSES
# ================================
courses_df = load_and_combine_courses("data")

# ================================
# JOB RANKING
# ================================
def rank_jobs(resume_text, top_n=5):

    resume_skills = extract_skills_light(resume_text)

    resume_desc_vector = desc_vectorizer.transform([resume_text])
    resume_title_vector = title_vectorizer.transform([resume_text])

    desc_scores = cosine_similarity(resume_desc_vector, desc_matrix)[0]
    title_scores = cosine_similarity(resume_title_vector, title_matrix)[0]

    scored = []

    for idx, row in jobs_df.iterrows():
        job_skills = row["skills"]

        if job_skills:
            overlap = len(resume_skills.intersection(job_skills)) / len(job_skills)
        else:
            overlap = 0.0

        final_score = (
            (0.55 * desc_scores[idx]) +
            (0.20 * title_scores[idx]) +
            (0.25 * overlap)
        )

        scored.append((idx, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n], resume_skills

# ================================
# COURSE RECOMMENDATION
# ================================
def recommend_courses(missing_skills, courses_df, top_n=2):
    recommendations = defaultdict(list)

    for skill in missing_skills:
        matched_df = courses_df[
            courses_df["skills"].apply(lambda x: skill in str(x).lower())
        ].copy()

        if matched_df.empty:
            recommendations[skill] = []
            continue

        matched_df = matched_df.sort_values(
            by=["rating", "reviewcount"],
            ascending=[False, False]
        )

        recommendations[skill] = (
            matched_df[["course", "rating", "level", "platform"]]
            .head(top_n)
            .to_dict(orient="records")
        )

    return recommendations

# ================================
# ROUTE
# ================================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        resume_text = request.form["resume"]

        ranked_jobs, resume_skills = rank_jobs(resume_text)
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

        result = {
            "job": best_job,
            "score": round(top_score * 100, 1),
            "matched": sorted(matched),
            "missing": sorted(missing),
            "courses": dict(course_recommendations),
            "top_matches": top_matches
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)