import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib, json
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

# =============================================================
# CONFIG
# =============================================================
st.set_page_config(
    page_title="Crop Recommendation Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

MAROON      = "#800000"
GOLD        = "#c9a84c"
GRAY        = "#6b6b6b"

CROP_COLORS = {
    "Rice"     : "#1f77b4", "Wheat"    : "#ff7f0e",
    "Millet"   : "#2ca02c", "Sugarcane": "#d62728",
    "Barley"   : "#9467bd", "Potato"   : "#8c564b",
    "Cotton"   : "#e377c2", "Pulses"   : "#7f7f7f",
    "Tomato"   : "#bcbd22", "Maize"    : "#17becf",
}

CROP_EMOJI = {
    "Rice":"🌾","Wheat":"🌿","Millet":"🌱","Sugarcane":"🎋",
    "Barley":"🌾","Potato":"🥔","Cotton":"🌸","Pulses":"🫘",
    "Tomato":"🍅","Maize":"🌽",
}

EKSPERIMEN_INFO = {
    "Eks 1 — DT Baseline": {
        "key":"eks_1_dt_baseline","fitur":"all","smote":False,
        "desc":"DT tanpa feature selection & SMOTE. Semua 20 fitur.",
        "accuracy":0.7990,"precision":0.8194,"recall":0.7990,"f1":0.8085,
    },
    "Eks 2 — DT + Chi2": {
        "key":"eks_2_dt_chi2","fitur":"chi2","smote":False,
        "desc":"DT + Chi-Square feature selection (top 10). Tanpa SMOTE.",
        "accuracy":0.7980,"precision":0.7954,"recall":0.7980,"f1":0.7960,
    },
    "Eks 3 — DT + RFE": {
        "key":"eks_3_dt_rfe","fitur":"rfe","smote":False,
        "desc":"DT + RFE feature selection (top 10). Tanpa SMOTE.",
        "accuracy":0.7910,"precision":0.7817,"recall":0.7910,"f1":0.7812,
    },
    "Eks 4 — DT + SMOTE": {
        "key":"eks_4_dt_smote","fitur":"all","smote":True,
        "desc":"DT tanpa feature selection + SMOTE (k=3). Data 29.576 baris.",
        "accuracy":0.7145,"precision":0.7328,"recall":0.7145,"f1":0.7206,
    },
    "Eks 5 — DT + Chi2 + SMOTE": {
        "key":"eks_5_dt_chi2_smote","fitur":"chi2","smote":True,
        "desc":"DT + Chi2 (top 10) + SMOTE. Kombinasi terbaik kedua.",
        "accuracy":0.7359,"precision":0.7587,"recall":0.7359,"f1":0.7419,
    },
    "Eks 6 — DT + RFE + SMOTE": {
        "key":"eks_6_dt_rfe_smote","fitur":"rfe","smote":True,
        "desc":"DT + RFE (top 10) + SMOTE.",
        "accuracy":0.6819,"precision":0.7018,"recall":0.6819,"f1":0.6893,
    },
}

CHI2_FEATURES = [
    "Rainfall","Temperature","Soil_pH_Bin","Soil_pH","N",
    "Soil_Type","Altitude_Bin","Wind_Speed","Irrigation_Type","Season"
]
RFE_FEATURES = [
    "N","K","Soil_pH","Soil_Moisture","Soil_Type",
    "Temperature","Rainfall","Sunlight_Hours","Wind_Speed","Fertilizer_Used"
]

# =============================================================
# STYLE — Solid Maroon Theme & Consistent Typography
# =============================================================
st.markdown(f"""
<style>
  /* ── sidebar ── */
  [data-testid="stSidebar"] {{
      background: {MAROON};
  }}
  [data-testid="stSidebar"] * {{ color: #fff !important; }}
  [data-testid="stSidebar"] .stRadio label {{ color: #fff !important; }}

  /* ── header text consistency ── */
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
      color: {MAROON} !important;
      font-weight: 700 !important;
  }}
  /* pastikan text header di sidebar & hero-header tetap putih */
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  .hero-header h1, .hero-header h2, .hero-header h3 {{
      color: white !important;
  }}

  /* ── hero header strip ── */
  .hero-header {{
      background: {MAROON};
      color: white;
      padding: 28px 32px;
      border-radius: 12px;
      margin-bottom: 24px;
  }}
  .hero-header h1 {{ margin: 0 0 6px 0; font-size: 26px; }}
  .hero-header p  {{ color: #f5c6c6; margin: 0; font-size: 14px; opacity: 0.9; }}

  /* ── metric card ── */
  .metric-card {{
      background: white;
      border-radius: 10px;
      padding: 18px 20px;
      border-left: 5px solid {MAROON};
      box-shadow: 0 2px 8px rgba(0,0,0,.08);
      margin-bottom: 10px;
  }}
  .metric-label {{ font-size: 12px; color: {GRAY}; text-transform: uppercase; font-weight: 600; }}
  .metric-value {{ font-size: 28px; font-weight: 700; color: {MAROON}; }}

  /* ── experiment card ── */
  .eks-card {{
      background: #fff8f8;
      border: 1px solid #f0d0d0;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 8px;
  }}
  .eks-best {{ border-left: 5px solid {MAROON}; background: #fff0f0; }}

  /* ── section title override for custom dividers ── */
  .section-title {{
      font-size: 18px; font-weight: 700;
      color: {MAROON}; margin-bottom: 12px;
      border-bottom: 2px solid {MAROON}; padding-bottom: 4px;
  }}

  /* ── badge ── */
  .badge-best {{
      background: {MAROON}; color: white;
      padding: 2px 10px; border-radius: 20px;
      font-size: 11px; font-weight: bold;
  }}
  .badge-smote {{
      background: #2ca02c; color: white;
      padding: 2px 8px; border-radius: 20px; font-size: 11px;
  }}
  .badge-no-smote {{
      background: {GRAY}; color: white;
      padding: 2px 8px; border-radius: 20px; font-size: 11px;
  }}

  /* ── prediction result ── */
  .pred-box {{
      background: {MAROON};
      color: white; border-radius: 14px;
      padding: 28px; text-align: center; margin: 16px 0;
  }}
  .pred-crop {{ font-size: 42px; font-weight: 800; margin: 8px 0; color: white !important;}}
  .pred-sub  {{ font-size: 14px; color: #f5c6c6; }}

  div[data-testid="stMetric"] {{
      background: white;
      border-radius: 8px;
      padding: 12px 16px;
      border-left: 4px solid {MAROON};
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}
</style>
""", unsafe_allow_html=True)

# =============================================================
# LOAD DATA & ARTIFACTS
# =============================================================
@st.cache_data
def load_data():
    p = Path("crop_recommendation.csv")
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_artifacts():
    art = {}
    base = Path(".")
    simple_files = {
        "scaler"          : "scaler.joblib",
        "label_encoder"   : "label_encoder_target.joblib",
        "feature_encoders": "feature_encoders.joblib",
        "feat_all"        : "feature_names_all.joblib",
        "feat_chi2"       : "feature_names_chi2.joblib",
        "feat_rfe"        : "feature_names_rfe.joblib",
    }
    for k, fname in simple_files.items():
        p = base / fname
        art[k] = joblib.load(p) if p.exists() else None

    meta = base / "feature_metadata.json"
    art["feature_meta"] = json.loads(meta.read_text()) if meta.exists() else {}

    art["models"] = {}
    for info in EKSPERIMEN_INFO.values():
        p = base / f"{info['key']}.joblib"
        if p.exists():
            art["models"][info["key"]] = joblib.load(p)
    return art

df   = load_data()
art  = load_artifacts()

dist = None
if df is not None:
    dist = df["Recommended_Crop"].value_counts().reset_index()
    dist.columns = ["Kelas", "Jumlah"]
    dist["Proporsi (%)"] = (dist["Jumlah"] / dist["Jumlah"].sum() * 100).round(2)

# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.markdown("## Crop Dashboard")
st.sidebar.markdown("Precision Agriculture · UMM 2025")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigasi", [
    "Overview",
    "Eksplorasi Data",
    "Performa Model",
    "Prediksi Tanaman",
    "Batch Prediksi",
    "Tentang Proyek",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Algoritma:** Decision Tree")
st.sidebar.markdown("**Eksperimen:** 6 Variasi Pipeline")
st.sidebar.markdown("**Model Terbaik:** DT Baseline")
st.sidebar.markdown("**Best F1-Score:** 0.8085")
st.sidebar.markdown("*(overall weighted avg)*")

# =============================================================
# PAGE 1 — OVERVIEW
# =============================================================
if page == "Overview":
    st.markdown("""
    <div class="hero-header">
      <h1>Crop Recommendation — Data Mining</h1>
      <p>Klasifikasi rekomendasi tanaman berbasis kondisi lahan & iklim dengan Decision Tree · Universitas Muhammadiyah Malang 2025</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset",     "10.000 baris")
    col2.metric("Kelas Target", "10 tanaman")
    col3.metric("Algoritma",   "Decision Tree")
    col4.metric("Eksperimen",  "6 variasi")

    st.markdown("---")
    st.markdown('<p class="section-title">Ringkasan 6 Eksperimen</p>', unsafe_allow_html=True)

    rows = []
    best_f1 = max(v["f1"] for v in EKSPERIMEN_INFO.values())
    for name, info in EKSPERIMEN_INFO.items():
        is_best = info["f1"] == best_f1
        smote_badge = '<span class="badge-smote">SMOTE</span>' if info["smote"] else '<span class="badge-no-smote">No SMOTE</span>'
        best_badge  = '<span class="badge-best">TERBAIK</span>' if is_best else ""
        st.markdown(f"""
        <div class="eks-card {'eks-best' if is_best else ''}">
          <b>{name}</b> {best_badge} &nbsp; {smote_badge} &nbsp;
          <span style="color:{GRAY};font-size:13px">Fitur: {'Top 10 ' + info['fitur'].upper() if info['fitur'] != 'all' else 'Semua (20)'}</span><br>
          <span style="font-size:13px;color:#333">{info['desc']}</span><br>
          <span style="font-size:13px">
            Acc <b>{info['accuracy']:.4f}</b> &nbsp;|&nbsp;
            Prec <b>{info['precision']:.4f}</b> &nbsp;|&nbsp;
            Recall <b>{info['recall']:.4f}</b> &nbsp;|&nbsp;
            F1 <b style="color:{MAROON}">{info['f1']:.4f}</b>
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Grafik Perbandingan F1-Score</p>', unsafe_allow_html=True)

    eks_labels = [k.split("—")[1].strip() for k in EKSPERIMEN_INFO.keys()]
    f1_vals    = [v["f1"] for v in EKSPERIMEN_INFO.values()]
    bar_colors = [MAROON if v == best_f1 else "#cc8888" for v in f1_vals]

    fig = go.Figure(go.Bar(
        x=eks_labels, y=f1_vals,
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in f1_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="F1-Score 6 Eksperimen Decision Tree",
        yaxis=dict(range=[0, 1.05], title="F1-Score"),
        xaxis_title="Eksperimen",
        plot_bgcolor="white",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Key Findings</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.info("**DT Baseline** overall F1 tertinggi (0.8085) — namun macro avg F1 hanya 0.47. Kelas minoritas (Tomato, Maize, Pulses) nyaris tidak bisa diprediksi.")
        st.success("**SMOTE meningkatkan fairness**: Tomato 0.00→0.68, Maize 0.06→0.63, Pulses 0.10→0.65 — dengan mengorbankan sedikit overall accuracy.")
    with c2:
        st.warning("**Chi2 > RFE** untuk DT. Selisih F1: +0.0148 (tanpa SMOTE) dan +0.0526 (dengan SMOTE).")
        st.error("**Fitur iklim dominan**: Rainfall & Temperature adalah root node & level 2 pohon keputusan — konsisten dengan Chi2 score tertinggi.")

# =============================================================
# PAGE 2 — EKSPLORASI DATA
# =============================================================
elif page == "Eksplorasi Data":
    st.markdown("""
    <div class="hero-header">
      <h1>Eksplorasi Data</h1>
      <p>Analisis distribusi dan karakteristik dataset Crop Recommendation</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("File `crop_recommendation.csv` tidak ditemukan.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Distribusi Kelas", "Korelasi", "Feature Analysis"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris",   f"{df.shape[0]:,}")
        col2.metric("Jumlah Kolom",   df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        st.subheader("10 Data Pertama")
        st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        st.subheader("Distribusi Kelas Target (Class Imbalance)")
        fig = px.bar(
            dist, x="Kelas", y="Jumlah",
            color="Jumlah",
            color_continuous_scale=[[0,"#f5c6c6"],[1, MAROON]],
            text="Jumlah",
            title="Distribusi Kelas Recommended_Crop",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, plot_bgcolor="white",
                          coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(dist, use_container_width=True, hide_index=True)
        with col2:
            ratio = dist.iloc[0]["Jumlah"] / dist.iloc[-1]["Jumlah"]
            st.warning(f"**Class Imbalance**: Rasio {ratio:.1f}x\n\n"
                       f"Dominan: **{dist.iloc[0]['Kelas']}** ({dist.iloc[0]['Jumlah']} data)\n\n"
                       f"Minoritas: **{dist.iloc[-1]['Kelas']}** ({dist.iloc[-1]['Jumlah']} data)")
            fig2 = px.pie(dist, names="Kelas", values="Jumlah",
                          color_discrete_sequence=list(CROP_COLORS.values()),
                          title="Proporsi Kelas")
            fig2.update_layout(height=340)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Heatmap Korelasi Fitur Numerik")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        aspect="auto", title="Heatmap Korelasi")
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Chi2 Feature Importance Score")
        chi2_data = {
            "Fitur"     : ["Rainfall","Temperature","Soil_pH_Bin","Soil_pH","N",
                           "Soil_Type","Altitude_Bin","Wind_Speed","Irrigation_Type","Season"],
            "Chi2 Score": [791.65,723.58,376.82,164.19,151.41,7.51,3.43,3.39,2.79,2.77],
            "Signifikan": ["Ya","Ya","Ya","Ya","Ya","Tidak","Tidak","Tidak","Tidak","Tidak"],
        }
        df_chi2 = pd.DataFrame(chi2_data)
        fig = px.bar(df_chi2, x="Chi2 Score", y="Fitur",
                     orientation="h",
                     color="Chi2 Score",
                     color_continuous_scale=[[0,"#f5c6c6"],[1, MAROON]],
                     title="Chi2 Score per Fitur (Top 10)")
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          yaxis=dict(categoryorder="total ascending"), height=380)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
# PAGE 3 — PERFORMA MODEL
# =============================================================
elif page == "Performa Model":
    st.markdown("""
    <div class="hero-header">
      <h1>Perbandingan 6 Eksperimen Decision Tree</h1>
      <p>Perbandingan Decision Tree dengan variasi Feature Selection (Chi2, RFE) dan Balancing (SMOTE)</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Grafik Metrik", "Detail Per Eksperimen", "Analisis Trade-off"])

    with tab1:
        eks_labels = [k.split("—")[1].strip() for k in EKSPERIMEN_INFO.keys()]
        best_f1    = max(v["f1"] for v in EKSPERIMEN_INFO.values())

        for metric, key in [("Accuracy","accuracy"),("Precision","precision"),
                             ("Recall","recall"),("F1-Score","f1")]:
            vals = [v[key] for v in EKSPERIMEN_INFO.values()]
            colors = [MAROON if v == max(vals) else "#cc8888" for v in vals]
            fig = go.Figure(go.Bar(
                x=eks_labels, y=vals,
                marker_color=colors,
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{metric} — 6 Eksperimen",
                yaxis=dict(range=[0, 1.1], title=metric),
                plot_bgcolor="white", height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        selected = st.selectbox("Pilih Eksperimen", list(EKSPERIMEN_INFO.keys()))
        info = EKSPERIMEN_INFO[selected]
        feat_list = CHI2_FEATURES if info["fitur"]=="chi2" else (RFE_FEATURES if info["fitur"]=="rfe" else (art.get("feat_all") or []))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{info['accuracy']:.4f}")
        col2.metric("Precision", f"{info['precision']:.4f}")
        col3.metric("Recall",    f"{info['recall']:.4f}")
        col4.metric("F1-Score",  f"{info['f1']:.4f}")

        st.info(f"{info['desc']}")
        st.markdown(f"""
        | Parameter | Nilai |
        |---|---|
        | Algoritma | Decision Tree (Gini Impurity, max_depth=10) |
        | Feature Selection | {'Top 10 ' + info['fitur'].upper() if info['fitur'] != 'all' else 'Semua fitur (20)'} |
        | Jumlah Fitur | {len(feat_list)} |
        | SMOTE | {'Ya — 29.576 baris (k=3)' if info['smote'] else 'Tidak — 8.000 baris'} |
        """)

        if feat_list:
            st.markdown("**Fitur yang digunakan:**")
            cols = st.columns(5)
            for i, f in enumerate(feat_list):
                cols[i % 5].markdown(f"- `{f}`")

    with tab3:
        st.subheader("Trade-off: Overall Accuracy vs Fairness Antar Kelas")

        # Scatter plot
        rows = []
        for name, info in EKSPERIMEN_INFO.items():
            rows.append({
                "Eksperimen": name.split("—")[1].strip(),
                "F1-Score (Overall)": info["f1"],
                "SMOTE": "Ya" if info["smote"] else "Tidak",
                "Feature Selection": info["fitur"].upper() if info["fitur"] != "all" else "Semua",
            })
        df_scatter = pd.DataFrame(rows)
        fig = px.scatter(df_scatter, x="F1-Score (Overall)", y="Eksperimen",
                         color="SMOTE", size=[30]*6,
                         color_discrete_map={"Ya":"#2ca02c","Tidak":MAROON},
                         title="F1-Score vs Konfigurasi SMOTE")
        fig.update_layout(plot_bgcolor="white", height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Perubahan F1 Kelas Minoritas: Baseline vs SMOTE")
        minority_data = {
            "Kelas"  : ["Barley","Potato","Cotton","Pulses","Tomato","Maize"],
            "F1 Baseline (Eks1)" : [0.18, 0.26, 0.48, 0.10, 0.00, 0.06],
            "F1 + SMOTE (Eks4)"  : [0.49, 0.56, 0.69, 0.65, 0.68, 0.63],
        }
        df_minor = pd.DataFrame(minority_data)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="DT Baseline", x=df_minor["Kelas"],
                              y=df_minor["F1 Baseline (Eks1)"],
                              marker_color="#cc8888"))
        fig2.add_trace(go.Bar(name="DT + SMOTE", x=df_minor["Kelas"],
                              y=df_minor["F1 + SMOTE (Eks4)"],
                              marker_color=MAROON))
        fig2.update_layout(barmode="group", plot_bgcolor="white",
                           yaxis=dict(range=[0,1.0], title="F1-Score"),
                           title="Perubahan F1 Kelas Minoritas setelah SMOTE", height=360)
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================
# PAGE 4 — PREDIKSI TANAMAN
# =============================================================
elif page == "Prediksi Tanaman":
    st.markdown("""
    <div class="hero-header">
      <h1>Prediksi Rekomendasi Tanaman</h1>
      <p>Masukkan kondisi lahan untuk mendapatkan rekomendasi tanaman secara otomatis</p>
    </div>
    """, unsafe_allow_html=True)

    if not art.get("scaler") or not art.get("label_encoder"):
        st.error("Artifacts tidak lengkap. Jalankan notebook Colab terlebih dahulu.")
        st.stop()

    selected_eks = st.selectbox("Pilih Eksperimen", list(EKSPERIMEN_INFO.keys()))
    info = EKSPERIMEN_INFO[selected_eks]
    model_key = info["key"]

    if model_key not in (art.get("models") or {}):
        st.warning(f"Model untuk {selected_eks} belum tersedia.")
        st.stop()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Nutrisi Tanah")
        N           = st.slider("Nitrogen (N) mg/kg", 20, 159, 90)
        P           = st.slider("Fosfor (P) mg/kg", 10, 89, 50)
        K           = st.slider("Kalium (K) mg/kg", 10, 119, 64)
        soil_ph     = st.slider("Soil pH", 4.5, 8.5, 6.5)
        soil_moist  = st.slider("Soil Moisture (%)", 10.0, 60.0, 35.0)
        organic_c   = st.slider("Organic Carbon", 0.2, 1.5, 0.85)
        elec_cond   = st.slider("Electrical Conductivity (dS/m)", 0.1, 3.0, 1.55)

    with col2:
        st.subheader("Kondisi Iklim")
        temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
        humidity    = st.slider("Humidity (%)", 30.0, 90.0, 60.0)
        rainfall    = st.slider("Rainfall (mm/tahun)", 200.0, 3000.0, 1500.0)
        sunlight    = st.slider("Sunlight Hours (jam/hari)", 4.0, 12.0, 8.0)
        wind_speed  = st.slider("Wind Speed (km/h)", 1.0, 20.0, 10.0)

    with col3:
        st.subheader("Manajemen Lahan")
        soil_type   = st.selectbox("Soil Type",        ["Clay","Loamy","Sandy","Silt"])
        region      = st.selectbox("Region",           ["Central","East","North","South","West"])
        altitude    = st.slider("Altitude (mdpl)", 0, 2499, 1240)
        season      = st.selectbox("Season",           ["Kharif","Rabi","Zaid"])
        irrigation  = st.selectbox("Irrigation Type",  ["Canal","Drip","Rainfed","Sprinkler"])
        fertilizer  = st.slider("Fertilizer Used (kg/ha)", 50.0, 350.0, 200.0)

    st.markdown("---")

    if st.button("Prediksi Tanaman", type="primary", use_container_width=True):
        enc = art["feature_encoders"]
        le  = art["label_encoder"]
        feat_all_names = art["feat_all"]

        soil_type_enc  = list(enc["Soil_Type"].classes_).index(soil_type)
        region_enc     = list(enc["Region"].classes_).index(region)
        season_enc     = list(enc["Season"].classes_).index(season)
        irrigation_enc = list(enc["Irrigation_Type"].classes_).index(irrigation)
        ph_bin   = "Acidic" if soil_ph<=6.0 else ("Neutral" if soil_ph<=7.0 else "Alkaline")
        ph_bin_enc   = list(enc["Soil_pH_Bin"].classes_).index(ph_bin)
        alt_bin  = "Rendah" if altitude<833 else ("Sedang" if altitude<1666 else "Tinggi")
        alt_bin_enc  = list(enc["Altitude_Bin"].classes_).index(alt_bin)

        row_all = pd.DataFrame([{
            "N":N,"P":P,"K":K,"Soil_pH":soil_ph,"Soil_Moisture":soil_moist,
            "Soil_Type":soil_type_enc,"Organic_Carbon":organic_c,
            "Electrical_Conductivity":elec_cond,"Temperature":temperature,
            "Humidity":humidity,"Rainfall":rainfall,"Sunlight_Hours":sunlight,
            "Wind_Speed":wind_speed,"Region":region_enc,"Altitude":altitude,
            "Season":season_enc,"Irrigation_Type":irrigation_enc,
            "Fertilizer_Used":fertilizer,"Soil_pH_Bin":ph_bin_enc,"Altitude_Bin":alt_bin_enc,
        }], columns=feat_all_names)

        row_scaled = pd.DataFrame(art["scaler"].transform(row_all), columns=feat_all_names)

        fitur_key = info["fitur"]
        feat_use  = CHI2_FEATURES if fitur_key=="chi2" else (RFE_FEATURES if fitur_key=="rfe" else feat_all_names)
        row_input = row_scaled[feat_use]

        model     = art["models"][model_key]
        pred_idx  = model.predict(row_input)[0]
        pred_crop = le.inverse_transform([pred_idx])[0]
        emoji     = CROP_EMOJI.get(pred_crop, "")

        proba = model.predict_proba(row_input)[0]
        top3_idx   = np.argsort(proba)[::-1][:3]
        top3_crops = le.inverse_transform(top3_idx)
        top3_proba = proba[top3_idx]

        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-sub">Tanaman yang Direkomendasikan</div>
          <div class="pred-crop">{emoji} {pred_crop}</div>
          <div class="pred-sub">Eksperimen: {selected_eks} · F1: {info['f1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Top 3 Prediksi")
        fig = go.Figure(go.Bar(
            x=top3_proba[::-1]*100,
            y=[f"{CROP_EMOJI.get(c,'')} {c}" for c in top3_crops[::-1]],
            orientation="h",
            marker_color=[MAROON, "#cc4444", "#e8a0a0"],
            text=[f"{v*100:.1f}%" for v in top3_proba[::-1]],
            textposition="outside",
        ))
        fig.update_layout(xaxis=dict(range=[0,115], title="Probabilitas (%)"),
                          plot_bgcolor="white", height=240)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================
# PAGE 5 — BATCH PREDIKSI
# =============================================================
elif page == "Batch Prediksi":
    st.markdown("""
    <div class="hero-header">
      <h1>Batch Prediksi</h1>
      <p>Upload CSV berisi data lahan untuk prediksi rekomendasi tanaman secara massal</p>
    </div>
    """, unsafe_allow_html=True)

    if not art.get("scaler") or not art.get("label_encoder"):
        st.error("Artifacts tidak lengkap.")
        st.stop()

    selected_eks = st.selectbox("Pilih Eksperimen untuk Batch Prediksi", list(EKSPERIMEN_INFO.keys()))
    info      = EKSPERIMEN_INFO[selected_eks]
    model_key = info["key"]

    st.info(f"Menggunakan: **{selected_eks}** · F1: {info['f1']:.4f}")

    st.markdown("**Format CSV yang diperlukan** (tanpa kolom Previous_Crop & Recommended_Crop):")
    feat_all_names = art.get("feat_all") or []
    non_bin = [f for f in feat_all_names if "_Bin" not in f]
    st.code(", ".join(non_bin[:10]) + ", ...")

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.write(f"Berhasil membaca **{len(df_upload)} baris** data.")
            st.dataframe(df_upload.head(), use_container_width=True)

            if model_key not in (art.get("models") or {}):
                st.warning("Model tidak tersedia.")
            else:
                enc = art["feature_encoders"]
                le  = art["label_encoder"]
                feat_all_names = art["feat_all"]

                df_proc = df_upload.copy()
                for col in ["Soil_Type","Region","Season","Irrigation_Type"]:
                    if col in df_proc.columns:
                        df_proc[col] = enc[col].transform(df_proc[col].astype(str))

                # FIX: Mencegah Data Leakage & Out-of-bounds error dengan menggunakan np.inf
                df_proc["Soil_pH_Bin"] = pd.cut(
                    df_proc["Soil_pH"],
                    bins=[-np.inf, 6.0, 7.0, np.inf], 
                    labels=["Acidic", "Neutral", "Alkaline"]
                )
                df_proc["Soil_pH_Bin"] = enc["Soil_pH_Bin"].transform(df_proc["Soil_pH_Bin"].astype(str))

                df_proc["Altitude_Bin"] = pd.cut(
                    df_proc["Altitude"], 
                    bins=[-np.inf, 832.99, 1665.99, np.inf], 
                    labels=["Rendah", "Sedang", "Tinggi"]
                )
                df_proc["Altitude_Bin"] = enc["Altitude_Bin"].transform(df_proc["Altitude_Bin"].astype(str))

                available = [c for c in feat_all_names if c in df_proc.columns]
                X_batch = pd.DataFrame(art["scaler"].transform(df_proc[available]), columns=available)

                fitur_key = info["fitur"]
                feat_use  = CHI2_FEATURES if fitur_key=="chi2" else (RFE_FEATURES if fitur_key=="rfe" else feat_all_names)
                feat_use  = [f for f in feat_use if f in X_batch.columns]

                # Prediksi class & probabilitas
                model_obj = art["models"][model_key]
                preds = le.inverse_transform(model_obj.predict(X_batch[feat_use]))
                proba = model_obj.predict_proba(X_batch[feat_use])
                confidences = proba.max(axis=1) * 100
                
                df_upload["Rekomendasi_Tanaman"] = preds
                df_upload["Confidence (%)"] = confidences.round(2)
                df_upload["Emoji"] = [CROP_EMOJI.get(c,"") for c in preds]

                st.success(f"Prediksi selesai untuk {len(preds)} baris")
                # Menampilkan hasil dengan kolom Emoji, Rekomendasi, Confidence di depan
                st.dataframe(df_upload[["Emoji","Rekomendasi_Tanaman", "Confidence (%)"] + list(df_upload.columns[:-3])],
                             use_container_width=True)

                fig = px.histogram(df_upload, x="Rekomendasi_Tanaman",
                                   title="Distribusi Hasil Prediksi Batch",
                                   color="Rekomendasi_Tanaman",
                                   color_discrete_map=CROP_COLORS)
                fig.update_layout(plot_bgcolor="white", height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                csv_out = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button("Download Hasil Prediksi CSV", csv_out,
                                   file_name="hasil_prediksi.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# =============================================================
# PAGE 6 — TENTANG PROYEK
# =============================================================
elif page == "Tentang Proyek":
    st.markdown("""
    <div class="hero-header">
      <h1>Tentang Proyek</h1>
      <p>Crop Recommendation Classification Using Data Mining</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tim")
        st.markdown("""
        | NIM | Nama |
        |---|---|
        | 202310370311015 | Bukhary Kelian |
        | 202310370311014 | Moch. Luqman Hakim |
        """)
        st.subheader("Business Objective")
        st.markdown("""
        Membangun sistem klasifikasi berbasis Data Mining yang secara otomatis
        memprediksi jenis tanaman yang paling direkomendasikan berdasarkan kondisi
        tanah, iklim, dan manajemen lahan — membantu petani & penyuluh pertanian
        membuat keputusan tanam yang akurat dan efisien.
        """)

    with col2:
        st.subheader("Dataset")
        st.markdown("""
        | Atribut | Nilai |
        |---|---|
        | Nama | Crop Recommendation Dataset (Extended) |
        | Sumber | Kaggle — Precision Agriculture |
        | Jumlah Baris | 10.000 |
        | Jumlah Kolom | 20 |
        | Target | Recommended_Crop (10 kelas) |
        | Missing Values | 0 |
        | Class Imbalance | 24.5x (Rice vs Maize) |
        """)

    st.subheader("Pipeline Eksperimen")
    st.markdown("""
    | Tahap | Teknik |
    |---|---|
    | Data Cleaning | Drop Previous_Crop, cek missing/duplikat |
    | Binning | Soil_pH → Acidic/Neutral/Alkaline, Altitude → Rendah/Sedang/Tinggi |
    | Encoding | Label Encoding (Soil_Type, Region, Season, Irrigation_Type) |
    | Scaling | StandardScaler (semua fitur numerik) |
    | Feature Selection | Chi2 (top 10) dan RFE berbasis DT (top 10) |
    | Balancing | SMOTE k_neighbors=3 → 36.970 data |
    | Split | 80% Train / 20% Test (stratify=y) |
    | Algoritma | Decision Tree (Gini, max_depth=10, random_state=42) |
    """)

    st.subheader("Ringkasan Hasil")
    rows = []
    for name, info in EKSPERIMEN_INFO.items():
        rows.append({"Eksperimen": name,
                     "Accuracy": info["accuracy"],
                     "Precision": info["precision"],
                     "Recall": info["recall"],
                     "F1-Score": info["f1"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
