from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# loading the data
jobs_df = pd.read_csv("data/data.csv")
courses_df = pd.read_csv("data/Coursera.csv")

# clean the test
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

jobs_df['clean_description'] = jobs_df['Description'].apply(clean_text)

# tf-idf vectorization
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(jobs_df['clean_description'])

# list the skills
skill_list = [
    "python", "sql", "excel", "power bi",
    "communication", "reporting", "dashboard",
    "data analysis", "data visualization"
]

def extract_skills(text):
    found = set()
    for skill in skill_list:
        if skill in text:
            found.add(skill)
    return found

# route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        resume_text = request.form["resume"]
        clean_resume = clean_text(resume_text)

        resume_vector = vectorizer.transform([clean_resume])
        similarity_scores = cosine_similarity(resume_vector, tfidf_matrix)

        top_index = similarity_scores[0].argmax()
        best_job = jobs_df.iloc[top_index]["Job Title"]
        best_job_desc = jobs_df.iloc[top_index]["clean_description"]

        resume_skills = extract_skills(clean_resume)
        job_skills = extract_skills(best_job_desc)

        matched = resume_skills.intersection(job_skills)
        missing = job_skills.difference(resume_skills)

        result = {
            "job": best_job,
            "matched": matched,
            "missing": missing
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
