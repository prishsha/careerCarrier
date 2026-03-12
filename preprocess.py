import os
import pandas as pd
import re


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
# COURSERA PREPROCESSING
# ================================
def preprocess_coursera(df):

    df = df.copy()

    df["course"] = df["course"]

    # CLEAN skills properly
    df["skills"] = df["skills"].apply(
        lambda x: clean_text(str(x))
    )

    df["rating"] = pd.to_numeric(
        df.get("rating", 0),
        errors="coerce"
    ).fillna(0)

    df["reviewcount"] = pd.to_numeric(
        df.get("reviewcount", 0),
        errors="coerce"
    ).fillna(0)

    df["level"] = df.get("level", "Unknown")

    df["platform"] = "Coursera"

    return df[[
        "course",
        "skills",
        "rating",
        "reviewcount",
        "level",
        "platform"
    ]]


# ================================
# UDEMY PREPROCESSING
# ================================
def preprocess_udemy(df):

    df = df.copy()

    df["course"] = df["title"]

    # Use description as skill source
    df["skills"] = df["description"].apply(clean_text)

    df["rating"] = pd.to_numeric(
        df.get("rating", 0),
        errors="coerce"
    ).fillna(0)

    df["reviewcount"] = pd.to_numeric(
        df.get("reviewcount", 0),
        errors="coerce"
    ).fillna(0)

    df["level"] = df.get("level", "Unknown")

    df["platform"] = "Udemy"

    return df[[
        "course",
        "skills",
        "rating",
        "reviewcount",
        "level",
        "platform"
    ]]


# ================================
# EDX PREPROCESSING
# ================================
def preprocess_edx(df):

    df = df.copy()

    df["course"] = df["title"]

    # edX uses associatedskills column
    if "associatedskills" in df.columns:
        df["skills"] = df["associatedskills"]
    else:
        df["skills"] = ""

    # edX dataset does not usually have ratings
    df["rating"] = 0
    df["reviewcount"] = 0

    df["level"] = df.get("level", "Unknown")

    df["platform"] = "Edx"

    return df[[
        "course",
        "skills",
        "rating",
        "reviewcount",
        "level",
        "platform"
    ]]


# ================================
# MASTER LOADER
# ================================
def load_and_combine_courses(folder_path="data"):

    dfs = []

    for file in os.listdir(folder_path):

        if not file.endswith(".csv") or file == "data.csv":
            continue

        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path)

        filename = file.lower()

        if "coursera" in filename:
            df = preprocess_coursera(df)

        elif "udemy" in filename:
            df = preprocess_udemy(df)

        elif "edx" in filename:
            df = preprocess_edx(df)

        else:
            continue  # ignore unknown datasets

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)