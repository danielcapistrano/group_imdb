from preprocessing import JSONCountTransformer
import streamlit as st
import pandas as pd
import pickle

# Load data
df = pd.read_csv("credits_with_titles_sample.csv")
df = df[["title", "cast"]].dropna().drop_duplicates(subset=["title"])

# Load the trained pipeline
with open("crew_classifier_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# UI
st.title("🎬 Crew Size Category Predictor")
st.write("Select a movie to predict the size of its crew based on the cast list.")

# Movie selection
movie_title = st.selectbox("Choose a movie:", df["title"].sort_values().unique())

# Get cast for the selected movie
selected_row = df[df["title"] == movie_title].iloc[0]
cast_json = selected_row["cast"]

# Predict on button click
if st.button("Predict Crew Size Category"):
    input_df = pd.DataFrame([{"cast": cast_json}])
    prediction = pipeline.predict(input_df)[0]

    # Map prediction to label
    label_map = {
        0: "Small crew (0–10)",
        1: "Medium crew (11–30)",
        2: "Large crew (>30)"
    }
    label = label_map.get(prediction, "Unknown")

    st.success(f"Predicted Crew Size Category: **{label}**")
