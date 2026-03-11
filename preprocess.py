import os
import pandas as pd

def load_and_combine_courses(folder_path="data"):
    dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv") and file != "data.csv":

            df = pd.read_csv(os.path.join(folder_path, file))
            platform = os.path.splitext(file)[0].capitalize()

            df = df.copy()
            df["platform"] = platform

            # -------- STANDARDIZE COURSE COLUMN --------
            if "course" in df.columns:
                df["course"] = df["course"]
            elif "title" in df.columns:
                df["course"] = df["title"]
            else:
                df["course"] = None

            # -------- STANDARDIZE SKILLS COLUMN --------
            if "skills" in df.columns:
                df["skills"] = df["skills"]
            elif "description" in df.columns:
                df["skills"] = df["description"]
            else:
                df["skills"] = ""

            # -------- SAFE RATING HANDLING --------
            if "rating" in df.columns:
                df["rating"] = pd.to_numeric(
                    df["rating"], errors="coerce"
                ).fillna(0)
            else:
                df["rating"] = 0

            # -------- SAFE REVIEWCOUNT HANDLING --------
            if "reviewcount" in df.columns:
                df["reviewcount"] = pd.to_numeric(
                    df["reviewcount"], errors="coerce"
                ).fillna(0)
            else:
                df["reviewcount"] = 0

            # -------- LEVEL --------
            if "level" in df.columns:
                df["level"] = df["level"]
            else:
                df["level"] = "Unknown"

            df = df[[
                "course",
                "skills",
                "rating",
                "reviewcount",
                "level",
                "platform"
            ]]

            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)