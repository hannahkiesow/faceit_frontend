import os
import io
import requests
import streamlit as st
from PIL import Image
from theme import apply_theme

# ---------- Theme & Page ----------
apply_theme()
st.set_page_config(page_title="Face It", page_icon="🤖", layout="wide")


st.title("Face It: We've Got Feelings")
st.image("media/banner.png", use_container_width=False, width=600)




st.header("👋 Hey there!")
st.write("We trained a slightly over-confident AI that thinks it can read your emotions 😏.")
st.write(" ")
st.write(" ")

st.header("All you have to do is:")
st.write("📸 Upload a face pic(front-facing, clear photo)")
st.write("👉 Sit back and let our model take its best guess!")

# ---------- API URL setup ----------
BASE_URI = os.getenv("CLOUD_API_URI") or st.secrets.get("cloud_api_uri") or "http://localhost:8000"
if not BASE_URI.endswith("/"):
    BASE_URI += "/"
PRED_ENDPOINT = BASE_URI + "predict"

st.markdown("⚡ Tip: No sunglasses, masks, or ninja disguises 🥷 (our AI is good, but not *that* good...yet). 🤓")
st.write(" ")
st.write(" ")

# ---------- File uploader ----------
uploaded = st.file_uploader("Upload here!", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Preview", use_container_width=False)

    if st.button("Identify Emotion!"):
        img_bytes = uploaded.getvalue()
        content_type = uploaded.type or "image/jpeg"

        with st.spinner("🤖 Running the AI model… please wait!"):
            try:
                # call the API
                files = {"file": (uploaded.name or "image.jpg", img_bytes, content_type)}
                r = requests.post(PRED_ENDPOINT, files=files, timeout=30)
                r.raise_for_status()

                data = r.json()

                # FastAPI returns {label, confidence, probabilities}
                label = data.get("label") or data.get("emotion", "unknown")
                conf = data.get("confidence")
                probs = data.get("probabilities") or data.get("scores") or {}

                # --- Emotion styles: text color + translucent background ---
                EMOTION_STYLES = {
                    "happy":    {"fg": "#111111", "bg": "#FACC1533"},  # yellow bg
                    "sad":      {"fg": "#DBEAFE", "bg": "#3B82F633"},  # blue
                    "angry":    {"fg": "#FEE2E2", "bg": "#EF444433"},  # red
                    "fear":     {"fg": "#EDE9FE", "bg": "#8B5CF633"},  # violet
                    "surprise": {"fg": "#FCE7F3", "bg": "#EC489933"},  # pink
                    "disgust":  {"fg": "#ECFDF5", "bg": "#10B98133"},  # green
                    "neutral":  {"fg": "#F3F4F6", "bg": "#9CA3AF33"},  # gray
                }

                conf_txt = f" ({conf:.1%})" if isinstance(conf, (int, float)) else ""
                style = EMOTION_STYLES.get(label.lower(), {"fg": "#FFFFFF", "bg": "#FFFFFF22"})

                # Badge display
                badge_html = f"""
                <div style="
                    display:flex; align-items:center; justify-content:center;
                    margin: 12px 0 4px 0;
                ">
                  <span style="
                      background:{style['bg']};
                      color:{style['fg']};
                      padding:12px 18px;
                      border-radius:999px;
                      font-weight:700;
                      font-size:24px;
                      letter-spacing:0.3px;
                      box-shadow: 0 4px 12px rgba(0,0,0,0.25);
                      border:1px solid rgba(255,255,255,0.15);
                  ">
                    👩 Predicted Emotion: {label}{conf_txt}
                  </span>
                </div>
                """
                st.markdown(badge_html, unsafe_allow_html=True)

                # --- Show probabilities as bar + pie chart (top 3) ---
                if isinstance(probs, dict) and probs:
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import numpy as np

                    # Data
                    series = pd.Series(probs).sort_values(ascending=False).head(3)

                    # Theme
                    BG_COLOR = "#001D7E"
                    TEXT_COLOR = "white"
                    PALETTE = ["#7C3AED", "#EC4899", "#F59E0B"]
                    sns.set_theme(style="whitegrid", font_scale=1.0)

                    col1, col2 = st.columns(2)

                    # --- Bar chart ---
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG_COLOR)
                        ax.set_facecolor(BG_COLOR)

                        sns.barplot(
                            x=series.index,
                            y=series.values,
                            palette=PALETTE[:len(series)],
                            ax=ax,
                            edgecolor="none"
                        )
                        ax.set_ylim(0, 1)
                        ax.set_xlabel("")
                        ax.set_ylabel("Probability", color=TEXT_COLOR)
                        ax.set_title("Top 3 Predicted Emotions", color=TEXT_COLOR, pad=12)

                        ax.tick_params(colors=TEXT_COLOR)
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                        ax.grid(axis="y", linestyle=":", linewidth=0.8, color='white', alpha=0.2)

                        for p in ax.patches:
                            ax.annotate(f"{p.get_height():.1%}",
                                        (p.get_x() + p.get_width()/2, p.get_height()),
                                        ha="center", va="bottom", xytext=(0, 6),
                                        textcoords="offset points", color=TEXT_COLOR)

                        st.pyplot(fig, clear_figure=True)

                    # --- Donut pie ---
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=BG_COLOR)
                        ax2.set_facecolor(BG_COLOR)

                        wedges, texts, autotexts = ax2.pie(
                            series.values,
                            labels=series.index,
                            autopct="%.1f%%",
                            startangle=90,
                            pctdistance=0.78,
                            labeldistance=1.05,
                            colors=PALETTE[:len(series)],
                            textprops={"color": TEXT_COLOR}
                        )
                        centre = plt.Circle((0, 0), 0.55, fc=BG_COLOR)
                        ax2.add_artist(centre)
                        ax2.axis("equal")
                        ax2.set_title("Top 3 Predicted Emotions", color=TEXT_COLOR, pad=12)

                        st.pyplot(fig2, clear_figure=True)

                else:
                    st.json(data)  # fallback: show raw payload

            except requests.exceptions.RequestException as e:
                st.error(f"API call failed: {e}")

#st.caption(f"🔗 Working with API at: {PRED_ENDPOINT}")  # dev only, remove in prod
