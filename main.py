import joblib
import logging

df = joblib.load('df_cleaned.pkl')
cosine_sim = joblib.load('cosine_sim.pkl')

def recommend_songs(song_name, top_n=5):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]

    logging.info("✅ Top %d recommendations ready.", top_n)
    result_df = df[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."

    return result_df


import streamlit as st

st.set_page_config(
    page_title="Music Recommender System",
    page_icon="🎧",
    layout="centered"
)


st.title("🎶 Instant Music Recommender")

song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song)
        if recommendations is None:
            st.warning("Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)