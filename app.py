import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Steam Game Popularity Prediction",
    page_icon="🎮",
    layout="wide"
)

# =====================================================
# LOAD MODEL & DATASET
# =====================================================
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("steam_pipeline.pkl")
    return pipeline


@st.cache_data
def load_dataset():
    file_id = "1AUJDCuY9OKSdty2Vc9CE2robJSF0nu_y"
    output = "games_march2025_cleaned.csv"

    if not os.path.exists(output):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output,
            quiet=False
        )

    return pd.read_csv(output)


pipeline = load_pipeline()
df = load_dataset()

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

CURRENT_YEAR = 2026

df["release_year"] = df["release_date"].dt.year
df["game_age"] = CURRENT_YEAR - df["release_year"]

# Handle missing game_age
df["game_age"] = df["game_age"].fillna(0)

# =====================================================
# TRAINING COLUMNS (MUST MATCH TRAINING)
# =====================================================
training_columns = [
    'required_age',
    'price',
    'dlc_count',
    'windows',
    'mac',
    'linux',
    'metacritic_score',
    'achievements',
    'recommendations',
    'developers',
    'publishers',
    'categories',
    'genres',
    'user_score',
    'positive',
    'negative',
    'average_playtime_forever',
    'median_playtime_forever',
    'discount',
    'peak_ccu',
    'pct_pos_total',
    'num_reviews_total',
    'game_age'
]

# =====================================================
# LABEL MAPPING
# =====================================================
label_mapping = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High"
}

# =====================================================
# HEADER
# =====================================================
st.title("🎮 Steam Game Popularity Prediction")
st.markdown("""
Aplikasi ini memprediksi tingkat popularitas game pada platform Steam
menggunakan Machine Learning berdasarkan metadata game.
""")

# =====================================================
# SIDEBAR FILTER
# =====================================================
st.sidebar.header("Game Selection")

game_names = sorted(df["name"].dropna().unique())

selected_game = st.sidebar.selectbox(
    "Select Game",
    game_names
)

# =====================================================
# SELECT GAME ROW
# =====================================================
selected_rows = df[df["name"] == selected_game]

if len(selected_rows) == 0:
    st.error("Game tidak ditemukan.")
    st.stop()

selected_row = selected_rows.iloc[0]

# =====================================================
# MAIN CONTENT
# =====================================================
col1, col2 = st.columns([1, 2])

# ---------- LEFT ----------
with col1:
    st.subheader("Game Information")

    if "header_image" in df.columns:
        image_url = selected_row["header_image"]

        if pd.notna(image_url):
            st.image(image_url)

    st.write("### Basic Info")
    st.write(f"**Name:** {selected_row['name']}")
    st.write(f"**Price:** ${selected_row['price']}")
    st.write(f"**Required Age:** {selected_row['required_age']}")
    st.write(f"**Game Age:** {selected_row['game_age']} years")

# ---------- RIGHT ----------
with col2:
    st.subheader("Metadata")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("### Technical")
        st.write(f"Windows: {selected_row['windows']}")
        st.write(f"Mac: {selected_row['mac']}")
        st.write(f"Linux: {selected_row['linux']}")
        st.write(f"Achievements: {selected_row['achievements']}")
        st.write(f"DLC Count: {selected_row['dlc_count']}")

    with info_col2:
        st.write("### Popularity Metrics")
        st.write(f"Recommendations: {selected_row['recommendations']}")
        st.write(f"Peak CCU: {selected_row['peak_ccu']}")
        st.write(f"Positive Reviews: {selected_row['positive']}")
        st.write(f"Negative Reviews: {selected_row['negative']}")
        st.write(f"Metacritic Score: {selected_row['metacritic_score']}")

    st.write("### Category")
    st.write(f"Developer: {selected_row['developers']}")
    st.write(f"Publisher: {selected_row['publishers']}")
    st.write(f"Genres: {selected_row['genres']}")
    st.write(f"Categories: {selected_row['categories']}")

# =====================================================
# PREDICTION BUTTON
# =====================================================
st.markdown("---")

if st.button("Predict Popularity"):

    input_data = pd.DataFrame(
        [selected_row[training_columns]]
    )

    prediction = pipeline.predict(input_data)[0]

    popularity = label_mapping[prediction]

    st.subheader("Prediction Result")

    if popularity == "Very High":
        st.success(f"Predicted Popularity: {popularity}")
    elif popularity == "High":
        st.info(f"Predicted Popularity: {popularity}")
    elif popularity == "Medium":
        st.warning(f"Predicted Popularity: {popularity}")
    else:
        st.error(f"Predicted Popularity: {popularity}")

    # ============================================
    # PROBABILITY SCORE
    # ============================================
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_data)[0]

        prob_df = pd.DataFrame({
            "Popularity": list(label_mapping.values()),
            "Probability": probs
        })

        st.subheader("Prediction Probability")
        st.bar_chart(
            prob_df.set_index("Popularity")
        )

        st.dataframe(prob_df)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "Steam Game Popularity Prediction using Machine Learning"
)