"""Global Development Compass — K-Means country clustering, model embedded in-file.
Run: streamlit run app.py
"""
import base64, pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

st.set_page_config(page_title="Global Development Compass", page_icon="🧭", layout="wide")

# ---- Model attached as base64 (no external .pkl file needed) --------------------------
MODEL_B64 = (
    "gASVOgkAAAAAAAB9lCiMBW1vZGVslIwXc2tsZWFybi5jbHVzdGVyLl9rbWVhbnOUjAZLTWVhbnOUk5QpgZR9lCiMCm5fY2x1c3Rl"
    "cnOUSwOMBGluaXSUjAlrLW1lYW5zKyuUjAhtYXhfaXRlcpRNLAGMA3RvbJRHPxo24uscQy2MBm5faW5pdJRLCowHdmVyYm9zZZRL"
    "AIwMcmFuZG9tX3N0YXRllEsqjAZjb3B5X3iUiIwJYWxnb3JpdGhtlIwFbGxveWSUjBFmZWF0dXJlX25hbWVzX2luX5SMFm51bXB5"
    "Ll9jb3JlLm11bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SUk5SMBW51bXB5lIwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsJhZRo"
    "FowFZHR5cGWUk5SMAk84lImIh5RSlChLA4wBfJROTk5K/////0r/////Sz90lGKJXZQojApjaGlsZF9tb3J0lIwHZXhwb3J0c5SM"
    "BmhlYWx0aJSMB2ltcG9ydHOUjAZpbmNvbWWUjAlpbmZsYXRpb26UjApsaWZlX2V4cGVjlIwJdG90YWxfZmVylIwEZ2RwcJRldJRi"
    "jA5uX2ZlYXR1cmVzX2luX5RLCYwEX3RvbJRoE4wGc2NhbGFylJOUaB+MAmY4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0"
    "lGJDCC9DHOviNho/lIaUUpSMB19uX2luaXSUSwqMCl9hbGdvcml0aG2UaBGMCl9uX3RocmVhZHOUSwGMEGNsdXN0ZXJfY2VudGVy"
    "c1+UaBVoGEsAhZRoGoeUUpQoSwFLA0sJhpRoNolD2DAsnJ51euq/zLQcfX6k5D/e+tDk80bnP+HN8HDbZsg/4QHtQHW/9z/AGrCY"
    "8Ajfv8mqCSP0RfE/dUb8Jg5X6b/YsVHyHdv5P83qUbJzw/U/crdT84oA3L/JFRO7SPfDv+HyIkPUN8i/EDYuSwn75b/tO5LWLrzZ"
    "P4Hg3vrOg/S/mIwSXs/W9T9T7DY49FXjvxuG9PpUA9q/pzTUnsQ0oL9TSYlcdrvMvyIEb/LRvZg/ln1cpwEd0L/iVJYUVpSRvyHq"
    "jD6OTdA/R8HvpG4o27+aNDbO0q/Wv5R0lGKMD19uX2ZlYXR1cmVzX291dJRLA4wHbGFiZWxzX5RoFWgYSwCFlGgah5RSlChLAUun"
    "hZRoH4wCaTSUiYiHlFKUKEsDaDdOTk5K/////0r/////SwB0lGKJQpwCAAABAAAAAgAAAAIAAAABAAAAAgAAAAIAAAACAAAAAAAA"
    "AAAAAAACAAAAAgAAAAAAAAACAAAAAgAAAAIAAAAAAAAAAgAAAAEAAAACAAAAAgAAAAIAAAABAAAAAgAAAAAAAAACAAAAAQAAAAEA"
    "AAACAAAAAQAAAAAAAAACAAAAAQAAAAEAAAACAAAAAgAAAAIAAAABAAAAAQAAAAEAAAACAAAAAQAAAAIAAAAAAAAAAAAAAAAAAAAC"
    "AAAAAgAAAAIAAAACAAAAAQAAAAEAAAACAAAAAgAAAAAAAAAAAAAAAQAAAAEAAAACAAAAAAAAAAEAAAAAAAAAAgAAAAIAAAABAAAA"
    "AQAAAAIAAAABAAAAAgAAAAAAAAACAAAAAgAAAAIAAAABAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAACAAAAAgAAAAEAAAABAAAAAAAA"
    "AAIAAAABAAAAAgAAAAIAAAABAAAAAQAAAAIAAAACAAAAAAAAAAIAAAABAAAAAQAAAAIAAAACAAAAAQAAAAAAAAABAAAAAgAAAAIA"
    "AAACAAAAAgAAAAIAAAACAAAAAQAAAAIAAAABAAAAAgAAAAAAAAAAAAAAAQAAAAEAAAAAAAAAAgAAAAEAAAACAAAAAgAAAAIAAAAC"
    "AAAAAgAAAAAAAAAAAAAAAgAAAAIAAAABAAAAAgAAAAIAAAABAAAAAgAAAAIAAAABAAAAAAAAAAAAAAAAAAAAAgAAAAEAAAAAAAAA"
    "AAAAAAIAAAACAAAAAQAAAAIAAAAAAAAAAAAAAAIAAAABAAAAAgAAAAEAAAABAAAAAgAAAAIAAAACAAAAAgAAAAEAAAACAAAAAAAA"
    "AAAAAAAAAAAAAgAAAAIAAAACAAAAAgAAAAIAAAABAAAAAQAAAJR0lGKMCGluZXJ0aWFflEdAiftlPklkp4wHbl9pdGVyX5RLE4wQ"
    "X3NrbGVhcm5fdmVyc2lvbpSMBTEuOC4wlHVijAZzY2FsZXKUjBtza2xlYXJuLnByZXByb2Nlc3NpbmcuX2RhdGGUjA5TdGFuZGFy"
    "ZFNjYWxlcpSTlCmBlH2UKIwJd2l0aF9tZWFulIiMCHdpdGhfc3RklIiMBGNvcHmUiGgSaBVoGEsAhZRoGoeUUpQoSwFLCYWUaCKJ"
    "XZQoaCZoJ2goaCloKmgraCxoLWguZXSUYmgwSwmMD25fc2FtcGxlc19zZWVuX5RoM2g2QwgAAAAAAOBkQJSGlFKUjAVtZWFuX5Ro"
    "FWgYSwCFlGgah5RSlChLAUsJhZRoNolDSJPLeFKRIkNA3eVY7fKNRECczNngQ0MbQIxoa5DycUdAdSxlEiy+0ECBx9ynmCAfQFlf"
    "A2eQo1FA+D7zL26VB0CL05rtE1LJQJR0lGKMBHZhcl+UaBVoGEsAhZRoGoeUUpQoSwFLCYWUaDaJQ0gjBewPvEKZQKmJTa1ZV4dA"
    "Oa+GMu//HUBK26q5wTSCQDaZnfHgBLZBfOFyB4bEW0BuE93kWqdTQBwbqw1eOQJANr+xQl3ns0GUdJRijAZzY2FsZV+UaBVoGEsA"
    "hZRoGoeUUpQoSwFLCYWUaDaJQ0gzoDLnnxpEQNDqYsFuVDtAFq+Ir6foBUDtjzgxEiM4QKzk+8gQxdJA7d+8zPgTJUAZkHShprsh"
    "QAgLCGsgJvg/QNxHv2/Y0UCUdJRiaFRoVXVijA9mZWF0dXJlX2NvbHVtbnOUXZQoaCZoJ2goaCloKmgraCxoLWguZYwRY2x1c3Rl"
    "cl9sYWJlbF9tYXCUfZQoaDNoTkMEAQAAAJSGlFKUjCBVbmRlci1EZXZlbG9wZWQgKE5lZWRzIEFpZCBNb3N0KZRoM2hOQwQCAAAA"
    "lIaUUpSMCkRldmVsb3BpbmeUaDNoTkMEAAAAAJSGlFKUjAlEZXZlbG9wZWSUdXUu"
)

bundle = pickle.loads(base64.b64decode(MODEL_B64))
model, scaler = bundle["model"], bundle["scaler"]
FEATURES, LABEL_MAP = bundle["feature_columns"], bundle["cluster_label_map"]

COLORS = {"Under-Developed (Needs Aid Most)": "#E8604C", "Developing": "#E8A33D", "Developed": "#3FA796"}

META = {  # feature: (label, unit, min, max, step)
    "child_mort": ("Child Mortality", "per 1000", 0, 220, 1),
    "exports":    ("Exports", "% GDP", 0, 200, 1),
    "health":     ("Health Spend", "% GDP", 0, 20, 0.1),
    "imports":    ("Imports", "% GDP", 0, 200, 1),
    "income":     ("Net Income", "USD", 0, 130000, 100),
    "inflation":  ("Inflation", "% / yr", -10, 105, 0.5),
    "life_expec": ("Life Expectancy", "years", 30, 90, 0.1),
    "total_fer":  ("Fertility Rate", "children/woman", 0.5, 8, 0.1),
    "gdpp":       ("GDP / Capita", "USD", 0, 130000, 100),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter&display=swap');
.stApp{background:radial-gradient(circle at 15% 0%,#16323A,#0B1D26 45%,#081419);color:#F4F1EA;font-family:Inter,sans-serif}
section[data-testid="stSidebar"]{background:#0E2229} section[data-testid="stSidebar"] *{color:#F4F1EA}
h1,h2,h3,.hero{font-family:'Playfair Display',serif !important}
.hero{font-weight:800;font-size:2.4rem;margin-bottom:.1rem}
.sub{color:#9FB4BA;margin-bottom:1.3rem}
.eyebrow{text-transform:uppercase;letter-spacing:3px;font-size:.7rem;color:#E8A33D;font-weight:600}
.card{background:rgba(244,241,234,.05);border:1px solid rgba(244,241,234,.1);border-radius:14px;padding:1.2rem 1.5rem}
.badge{padding:.3rem 1rem;border-radius:999px;font-weight:700}
div[data-testid="stMetric"]{background:rgba(244,241,234,.05);border:1px solid rgba(244,241,234,.08);border-radius:12px;padding:.6rem}
.stButton>button{background:linear-gradient(135deg,#E8A33D,#E8604C);color:#0B1D26;border:none;font-weight:700;border-radius:10px}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_scored():
    df = pd.read_csv("Country-data.csv")
    Xs = scaler.transform(df[FEATURES])
    df["cluster"] = model.predict(Xs)
    df["category"] = df["cluster"].map(LABEL_MAP)
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    df[["pc1", "pc2"]] = pca.transform(Xs)
    return df, pca


df, pca = load_scored()

with st.sidebar:
    st.markdown("<div class='eyebrow'>Model Input</div>", unsafe_allow_html=True)
    st.markdown("### Build a Country Profile")
    pick = st.selectbox("Autofill from a country", ["— custom —"] + sorted(df["country"]))
    base = df[df["country"] == pick].iloc[0] if pick != "— custom —" else df[FEATURES].mean()
    st.markdown("---")
    vals = {}
    for f in FEATURES:
        label, unit, lo, hi, step = META[f]
        vals[f] = st.slider(f"{label} ({unit})", float(lo), float(hi),
                             float(min(max(base[f], lo), hi)), float(step))
    st.markdown("---")
    st.caption("K-Means (k=3) · scikit-learn · model embedded in this file")

st.markdown("<div class='eyebrow'>Unsupervised Learning · K-Means</div>", unsafe_allow_html=True)
st.markdown("<div class='hero'>Global Development Compass</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>See which development tier any country profile falls into, "
            "and where it lands among 167 real nations.</div>", unsafe_allow_html=True)

X = pd.DataFrame([vals])[FEATURES]
Xs = scaler.transform(X)
cluster = int(model.predict(Xs)[0])
category = LABEL_MAP[cluster]
color = COLORS[category]
pt = pca.transform(Xs)[0]

c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    st.markdown(f"""<div class='card'><div class='eyebrow'>Predicted Tier</div>
        <span class='badge' style='background:{color}22;color:{color};border:1px solid {color}66;'>
        ● {category}</span></div>""", unsafe_allow_html=True)
with c2:
    st.metric("GDP / Capita", f"${vals['gdpp']:,.0f}")
    st.metric("Child Mortality", f"{vals['child_mort']:.1f}/1000")
with c3:
    st.metric("Net Income", f"${vals['income']:,.0f}")
    st.metric("Life Expectancy", f"{vals['life_expec']:.1f} yrs")

st.markdown("<br>", unsafe_allow_html=True)
col_a, col_b = st.columns([1.5, 1])

with col_a:
    st.markdown("#### Where this profile sits among 167 countries")
    fig = px.scatter(df, x="pc1", y="pc2", color="category", color_discrete_map=COLORS,
                      hover_name="country", opacity=0.75)
    fig.add_scatter(x=[pt[0]], y=[pt[1]], mode="markers", name="Your Profile",
                     marker=dict(size=20, color=color, symbol="star", line=dict(color="#F4F1EA", width=2)))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font_color="#F4F1EA", legend_title_text="", height=420, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(gridcolor="rgba(244,241,234,.08)")
    fig.update_yaxes(gridcolor="rgba(244,241,234,.08)")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### Countries per tier")
    counts = df["category"].value_counts().reindex(list(COLORS))
    donut = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6,
                              marker=dict(colors=list(COLORS.values())), textinfo="value"))
    donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#F4F1EA", height=420,
                         showlegend=True, legend=dict(orientation="h", y=-0.2),
                         margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(donut, use_container_width=True)

st.markdown("#### Top 10 countries most in need of aid")
aid = (df[df["category"] == "Under-Developed (Needs Aid Most)"]
       .sort_values("gdpp")[["country", "child_mort", "income", "gdpp", "life_expec"]]
       .head(10).reset_index(drop=True))
aid.columns = ["Country", "Child Mortality", "Income", "GDP/Capita", "Life Expectancy"]
st.dataframe(aid.style.format({"Child Mortality": "{:.1f}", "Income": "${:,.0f}",
                                "GDP/Capita": "${:,.0f}", "Life Expectancy": "{:.1f}"})
             .background_gradient(subset=["GDP/Capita"], cmap="OrRd_r"),
             use_container_width=True, height=390)

st.caption("Model trained on 167 countries · 9 socio-economic & health indicators · adjust sliders to see live updates.")
