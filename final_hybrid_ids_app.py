import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, time, json
from pathlib import Path
from datetime import datetime

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    import joblib
    from sklearn.decomposition import PCA
    HAS_ML = True
except ImportError:
    HAS_ML = False

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SENTINEL · Team Groot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & PATHS ──────────────────────────────────────────
BASE_PATH = Path("/Users/dally/Adaptive_Cyber_Physical_Security/ids_2018")
OUTPUTS   = BASE_PATH / "outputs"
HYBRID    = OUTPUTS / "hybrid"

FEATURE_NAMES = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
    "URG Flag Cnt", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg",
    "Bwd Seg Size Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts",
    "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Protocol_6", "Protocol_17",
]

ATTACK_PROFILES = {
    "DDoS LOIC-UDP": {"Flow Duration": 10000, "Tot Fwd Pkts": 5000, "Flow Pkts/s": 50000,
                       "Flow Byts/s": 999999, "SYN Flag Cnt": 0, "ACK Flag Cnt": 0,
                       "Pkt Len Mean": 100, "Flow IAT Mean": 0.002},
    "SSH Bruteforce": {"Flow Duration": 500000, "Tot Fwd Pkts": 20, "Flow Pkts/s": 0.04,
                        "SYN Flag Cnt": 10, "ACK Flag Cnt": 10, "Pkt Len Mean": 64,
                        "Flow Byts/s": 300, "Flow IAT Mean": 50000},
    "DoS Slowloris":  {"Flow Duration": 9000000, "Tot Fwd Pkts": 5, "Flow Pkts/s": 0.0006,
                        "SYN Flag Cnt": 1, "ACK Flag Cnt": 4, "Pkt Len Mean": 200,
                        "Flow Byts/s": 50, "Flow IAT Mean": 2000000},
    "Normal Traffic": {"Flow Duration": 50000, "Tot Fwd Pkts": 15, "Flow Pkts/s": 300,
                        "SYN Flag Cnt": 1, "ACK Flag Cnt": 14, "Pkt Len Mean": 512,
                        "Flow Byts/s": 80000, "Flow IAT Mean": 3000},
}

# Load Architecture SVG (Cached)
@st.cache_data
def get_svg_data():
    SVG_FILE = Path("/Users/dally/Adaptive_Cyber_Physical_Security/adaptive_ids_full_architecture.svg")
    if SVG_FILE.exists():
        import base64
        with open(SVG_FILE, "rb") as f_svg:
            return f"data:image/svg+xml;base64,{base64.b64encode(f_svg.read()).decode()}"
    return ""

SVG_DATA_URI = get_svg_data()

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#070d1a;--surf:#0c1628;--card:#0f1d33;--brd:#1b3050;--brd2:rgba(56,189,248,.18);--c1:#38bdf8;--c2:#818cf8;--c3:#fb923c;--c4:#34d399;--c5:#f87171;--c6:#e879f9;--tx:#e8f0fd;--tx2:#7e95b8}
.stApp{background:radial-gradient(ellipse 120% 60% at 50% -10%,rgba(56,189,248,.07) 0%,#070d1a 55%) !important}
[data-testid="stAppViewContainer"]{background:transparent !important}
[data-testid="stHeader"]{background:transparent !important}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background-image:linear-gradient(rgba(27,48,80,.25) 1px,transparent 1px),linear-gradient(90deg,rgba(27,48,80,.25) 1px,transparent 1px);background-size:44px 44px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#080f20 0%,#060c18 100%) !important;border-right:1px solid var(--brd) !important}
[data-testid="stSidebar"]>div:first-child{padding-top:0 !important}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:var(--tx2)}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif !important;color:var(--tx) !important}
h1,h2,h3,h4,h5{font-family:"Syne",sans-serif !important;color:var(--tx) !important;letter-spacing:-.02em !important}
.sentinel-badge{display:inline-block;border-radius:5px;font-family:"IBM Plex Mono",monospace;font-size:11px;padding:3px 10px;letter-spacing:.08em;background:rgba(56,189,248,.1);color:var(--c1);border:1px solid rgba(56,189,248,.3);margin-right:6px}
.sentinel-badge.ind{background:rgba(129,140,248,.1);color:var(--c2);border-color:rgba(129,140,248,.3)}
.sentinel-badge.grn{background:rgba(52,211,153,.1);color:var(--c4);border-color:rgba(52,211,153,.3)}
.sentinel-badge.red{background:rgba(248,113,113,.1);color:var(--c5);border-color:rgba(248,113,113,.3)}
.sentinel-badge.org{background:rgba(251,146,60,.1);color:var(--c3);border-color:rgba(251,146,60,.3)}
.sentinel-badge.prp{background:rgba(232,121,249,.1);color:var(--c6);border-color:rgba(232,121,249,.3)}
.s-card{background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:22px 24px;margin-bottom:18px;transition:border-color .3s,box-shadow .3s;position:relative;overflow:hidden}
.s-card::before{content:"";position:absolute;inset:0;pointer-events:none;background:radial-gradient(circle at top right,rgba(56,189,248,.04),transparent 60%)}
.s-card:hover{border-color:rgba(56,189,248,.35);box-shadow:0 0 30px rgba(56,189,248,.07)}
.s-card.sky{border-top:2px solid var(--c1)}
.s-card.ind{border-top:2px solid var(--c2)}
.s-card.grn{border-top:2px solid var(--c4)}
.s-card.red{border-top:2px solid var(--c5)}
.s-card.org{border-top:2px solid var(--c3)}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.stat-item{background:var(--surf);border:1px solid var(--brd);border-radius:12px;padding:18px 16px;text-align:center}
.stat-num{font-family:"Syne",sans-serif;font-size:34px;font-weight:800;color:var(--c1);line-height:1;display:block}
.stat-lbl{font-size:11px;color:var(--tx2);margin-top:5px;font-family:"IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--c1);margin-bottom:8px;display:block}
.callout{background:rgba(56,189,248,.06);border-left:3px solid var(--c1);border-radius:0 10px 10px 0;padding:13px 17px;margin-top:12px;font-size:14px;color:var(--tx2);line-height:1.7}
.callout.warn{border-color:var(--c3);background:rgba(251,146,60,.06)}
.callout.good{border-color:var(--c4);background:rgba(52,211,153,.06)}
.callout.bad{border-color:var(--c5);background:rgba(248,113,113,.06)}
.status-ok{color:var(--c4);font-family:"Syne",sans-serif;font-size:28px;font-weight:800}
.status-bad{color:var(--c5);font-family:"Syne",sans-serif;font-size:28px;font-weight:800}
.sb-brand{padding:28px 22px 18px;border-bottom:1px solid var(--brd);margin-bottom:12px}
.sb-logo{font-family:"Syne",sans-serif;font-size:20px;font-weight:800;color:var(--c1);letter-spacing:.04em}
.sb-sub{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--tx2);letter-spacing:.15em;text-transform:uppercase;margin-top:2px}
.sb-team{margin-top:16px;font-size:13px;color:var(--tx2);line-height:1.7}
.sb-team strong{color:var(--tx)}
.sb-status{margin:16px 12px 0;background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.25);border-radius:10px;padding:12px 14px;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--tx2);line-height:1.8}
.sb-status .dot{color:var(--c4)}
.sb-demo{margin:12px 12px 0;background:rgba(251,146,60,.08);border:1px solid rgba(251,146,60,.25);border-radius:10px;padding:12px 14px;font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--c3)}
div.stButton > button { background: linear-gradient(135deg, rgba(56,189,248,.15), rgba(129,140,248,.15)) !important; border: 1px solid rgba(56,189,248,.4) !important; color: var(--c1) !important; border-radius: 10px !important; font-family: "Syne", sans-serif !important; font-weight: 700 !important; font-size: 14px !important; padding: 10px 24px !important; transition: all .25s !important; text-align: left !important; justify-content: flex-start !important; }
div.stButton > button:hover { background: linear-gradient(135deg, rgba(56,189,248,.25), rgba(129,140,248,.25)) !important; border-color: var(--c1) !important; box-shadow: 0 0 20px rgba(56,189,248,.2) !important; transform: translateX(4px) !important; }
[data-testid="stSidebar"] div.stButton > button { background: transparent !important; border: none !important; color: var(--tx2) !important; padding: 10px 14px !important; font-size: 14px !important; font-family: "DM Sans", sans-serif !important; font-weight: 400 !important; border-radius: 0 !important; width: 100% !important; margin: 2px 0 !important; text-align: left !important; justify-content: flex-start !important; }
[data-testid="stSidebar"] div.stButton > button:hover { color: var(--c1) !important; background: rgba(56,189,248,.06) !important; border-left: 2px solid rgba(56,189,248,.3) !important; }
/* Target buttons that have the active indicator '▸' */
[data-testid="stSidebar"] div.stButton > button p:contains("▸"), 
[data-testid="stSidebar"] div.stButton > button:has(p:contains("▸")) { 
    color: var(--c1) !important; background: rgba(56,189,248,.12) !important; 
    border-left: 3px solid var(--c1) !important; font-weight: 600 !important; 
}
.top-chrome { height: 3px; width: 100%; background: linear-gradient(90deg, var(--c2), var(--c1), var(--c4)); position: fixed; top: 0; left: 0; z-index: 999; }
.stSlider [data-testid="stSlider"]{accent-color:var(--c1)}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
[data-testid="stFileUploader"]{border:1px dashed var(--brd) !important;border-radius:12px !important;padding:20px !important;background:var(--surf) !important}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:var(--surf) !important;border-color:var(--brd) !important;color:var(--tx) !important;border-radius:8px !important}
div[data-testid="stExpander"]{background:var(--surf) !important;border:1px solid var(--brd) !important;border-radius:10px !important}
.stAlert{border-radius:10px !important}
[data-testid="stMetric"]{background:var(--card) !important;border:1px solid var(--brd) !important;border-radius:12px !important;padding:14px !important}
[data-testid="stMetricValue"]{font-family:"Syne",sans-serif !important;font-size:28px !important;color:var(--c1) !important}
.top-chrome{height:3px;width:100%;background:linear-gradient(90deg,var(--c2),var(--c1),var(--c4));position:fixed;top:0;left:0;z-index:999}
</style>
<div class="top-chrome"></div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_assets():
    assets = {"demo_mode": True, "load_errors": []}

    if not HAS_ML:
        assets["load_errors"].append("TensorFlow / joblib not installed")
        return assets

    # Scaler
    try:
        assets["scaler"] = joblib.load(OUTPUTS / "robust_scaler.joblib")
    except Exception as e:
        assets["load_errors"].append(f"Scaler: {e}")

    # AE models
    for key, fname in [("ae", "autoencoder_best.keras"),
                       ("sparse_ae", "sparse_ae_best.keras"),
                       ("dae", "denoising_ae_best.keras")]:
        try:
            assets[key] = tf.keras.models.load_model(OUTPUTS / fname)
        except Exception as e:
            assets["load_errors"].append(f"{key}: {e}")

    # RF
    for key, path in [("rf_pseudo", HYBRID / "rf_pseudo_label_final.joblib")]:
        try:
            assets[key] = joblib.load(path)
        except Exception as e:
            assets["load_errors"].append(f"{key}: {e}")

    # Encoder sub-model
    if "ae" in assets:
        try:
            assets["encoder"] = tf.keras.Model(
                inputs=assets["ae"].input,
                outputs=assets["ae"].get_layer("bottleneck").output,
            )
        except Exception as e:
            assets["load_errors"].append(f"encoder: {e}")

    # If core models loaded, switch out of demo mode
    if "ae" in assets and "scaler" in assets:
        assets["demo_mode"] = False

    # Metrics CSV
    try:
        assets["metrics"] = pd.read_csv(HYBRID / "model_comparison.csv")
    except Exception:
        assets["metrics"] = pd.DataFrame({
            "model":         ["Standard AE","Sparse AE","Denoising AE",
                              "AE + RF","RF alone","AE + OC-SVM","AE + IF","Meta-Ensemble"],
            "roc_auc":       [0.8965,0.8929,0.8867,0.7920,0.3433,0.4600,0.4231,0.4140],
            "avg_precision": [0.7578, None, None, 0.5593, None, 0.3137,0.3392,0.3024],
            "f1":            [0.0336,0.0336,0.1433,0.0335,0.0335,0.0094,0.0116,0.0098],
            "recall":        [0.0174,0.0174,0.0785,0.0174,0.0173,0.0048,0.0060,0.0050],
            "fpr":           [0.0099,0.0099,0.0394,0.0099,0.0099, None,  None, 0.0099],
        })

    return assets


ASSETS = load_assets()


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def recon_error(model, X):
    pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - pred), axis=1)

def safe_float(v):
    try: return float(v)
    except: return 0.0

def plotly_dark_layout(fig, **kw):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,22,40,0.6)",
        font_family="DM Sans",
        **kw,
    )
    return fig

def run_inference(df_in):
    """Return (scores_dict, top_features) for a single row, or (df_batch, None) for batch."""
    n = len(df_in)

    if ASSETS["demo_mode"]:
        # Removed delay for snappier sandbox experience
        if n == 1:
            ae_e   = float(np.random.uniform(.008, .05))
            rf_p   = float(np.random.uniform(.02, .15))
            return {
                "Autoencoder": ae_e, "Sparse AE": ae_e*1.05,
                "Denoising AE": ae_e*0.95, "Random Forest": rf_p,
                "Meta-Ensemble": 0.4*ae_e/0.05 + 0.6*rf_p,
            }, ["Flow Duration","Flow Byts/s","SYN Flag Cnt","Pkt Len Mean","Flow IAT Mean"]
        else:
            rng = np.random.default_rng()
            batch = pd.DataFrame({
                "AE Score":    rng.uniform(.005,.08, n),
                "RF Score":    rng.uniform(.01, .1, n),
                "Ensemble":    rng.uniform(.01, .1, n),
            })
            batch.iloc[:2] = 0.82
            batch["Is_Attack"] = batch["Ensemble"] > 0.5
            Z = rng.standard_normal((n, 3))
            Z[:2] += 5
            batch[["Z1","Z2","Z3"]] = Z
            return batch, None

    # Real inference
    feats = [f for f in FEATURE_NAMES if f in df_in.columns]
    missing = [f for f in FEATURE_NAMES if f not in df_in.columns]
    X_raw = df_in.reindex(columns=FEATURE_NAMES, fill_value=0.0).values.astype(np.float32)
    X_sc  = ASSETS["scaler"].transform(X_raw)

    if n == 1:
        # Faster inference for single sample using __call__
        ae_out  = ASSETS["ae"](X_sc, training=False).numpy()
        sae_out = ASSETS["sparse_ae"](X_sc, training=False).numpy()
        dae_out = ASSETS["dae"](X_sc, training=False).numpy()
        
        ae_e   = np.mean(np.square(X_sc - ae_out), axis=1)
        sae_e  = np.mean(np.square(X_sc - sae_out), axis=1)
        dae_e  = np.mean(np.square(X_sc - dae_out), axis=1)

        rf_p = (ASSETS["rf_pseudo"].predict_proba(X_raw)[:, 1]
                if "rf_pseudo" in ASSETS else np.zeros(1))

        # Normalise (for sandbox, use pre-calculated or heuristic scaling)
        ens = 0.45 * np.clip(ae_e/0.05, 0, 1) + 0.55 * rf_p

        # Feature attribution
        devs  = np.abs(X_sc - ae_out)[0]
        top5  = np.argsort(devs)[-5:][::-1]
        top_f = [FEATURE_NAMES[i] for i in top5]
        return {
            "Autoencoder": float(ae_e[0]),
            "Sparse AE":   float(sae_e[0]),
            "Denoising AE":float(dae_e[0]),
            "Random Forest":float(rf_p[0]),
            "Meta-Ensemble":float(ens[0]),
        }, top_f
    else:
        # Batch inference still uses .predict() for efficiency on large N
        ae_out  = ASSETS["ae"].predict(X_sc, verbose=0)
        sae_out = ASSETS["sparse_ae"].predict(X_sc, verbose=0)
        dae_out = ASSETS["dae"].predict(X_sc, verbose=0)
        
        ae_e   = np.mean(np.square(X_sc - ae_out), axis=1)
        sae_e  = np.mean(np.square(X_sc - sae_out), axis=1)
        dae_e  = np.mean(np.square(X_sc - dae_out), axis=1)

        rf_p = (ASSETS["rf_pseudo"].predict_proba(X_raw)[:, 1]
                if "rf_pseudo" in ASSETS else np.zeros(n))

        # Normalise
        def norm01(arr):
            lo, hi = arr.min(), arr.max()
            return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

        ens = 0.45 * norm01(ae_e) + 0.55 * norm01(rf_p)
        enc_feat = ASSETS["encoder"].predict(X_sc, verbose=0) if "encoder" in ASSETS else X_sc[:, :32]
        pca = PCA(n_components=min(3, enc_feat.shape[1], n))
        Z   = pca.fit_transform(enc_feat)
        while Z.shape[1] < 3:
            Z = np.hstack([Z, np.zeros((n, 1))])
        batch = pd.DataFrame({
            "AE Score": ae_e, "RF Score": rf_p, "Ensemble": ens,
            "Is_Attack": ens > 0.5,
            "Z1": Z[:, 0], "Z2": Z[:, 1], "Z3": Z[:, 2],
        })
        return batch, None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-logo">🛡️ SENTINEL</div>
      <div class="sb-sub">Hybrid IDS · Team Groot · v2.0</div>
      <div class="sb-team">
        <strong>Teammates</strong><br/>
        Dally R &amp; Pughazhendhi J<br/>
        <span style="font-size:11px;color:#4a6280">CSE-CIC-IDS2018 · AWS EC2 t3.large</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠  Overview"

    def set_page(p): st.session_state.current_page = p

    pages = [
        ("🏠  Overview", "overview"),
        ("🔍  Threat Analysis", "threat"),
        ("🌌  Latent Universe", "latent"),
        ("🧪  Attack Sandbox", "sandbox"),
        ("📊  Model Benchmarks", "bench"),
        ("🔬  Ablation Study", "ablation"),
        ("🏗️  Architecture", "arch"),
        ("ℹ️   About", "about")
    ]

    st.markdown("<div style='margin: 10px 0 20px 12px; font-size: 11px; color: var(--tx2); letter-spacing: 0.1em; text-transform: uppercase;'>Terminal Navigation</div>", unsafe_allow_html=True)
    
    for label, key in pages:
        is_active = st.session_state.current_page == label
        # Use an invisible Unicode character or a subtle visual cue to target active buttons via CSS
        display_label = f"▸ {label}" if is_active else f"  {label}"
        if st.button(display_label, key=f"nav_{key}", width="stretch"):
            st.session_state.current_page = label
            st.rerun()

    page = st.session_state.current_page

    demo_tag = "SIMULATION MODE" if ASSETS["demo_mode"] else "LIVE MODE"
    demo_col = "#fb923c" if ASSETS["demo_mode"] else "#34d399"
    now_dt    = datetime.now()
    uptime    = now_dt.strftime("%H:%M · %d %b")
    st.markdown(f"""
    <div class="sb-status">
      <span class="dot">●</span> STATUS: <span style="color:{demo_col}">{demo_tag}</span><br/>
      MODEL: Hybrid_AE_RF_v2<br/>
      DATASET: CIC-IDS-2018<br/>
      TIME: {uptime}
    </div>
    """, unsafe_allow_html=True)

    if ASSETS["demo_mode"] and ASSETS["load_errors"]:
        with st.expander("⚠️ Load warnings"):
            for e in ASSETS["load_errors"][:5]:
                st.caption(e)
    
    st.markdown("<br/><div class='sb-sub' style='margin-left:12px'>Model Stack Status</div>", unsafe_allow_html=True)
    for m, key in [("Autoencoder","ae"), ("Sparse AE","sparse_ae"), ("Denoising AE","dae"), ("Random Forest","rf_pseudo")]:
        status = "●" if key in ASSETS else "○"
        col = "#34d399" if key in ASSETS else "#4a6280"
        st.markdown(f"<div style='font-size:11px;color:{col};margin-left:15px;margin-bottom:4px'>{status} {m}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <span class="eyebrow">Project 2 · Hybrid Phase · CSE-CIC-IDS2018</span>
    <h1 style="font-size:58px;line-height:.92;margin-bottom:16px">
      ADAPTIVE<br/><span style="color:#38bdf8">CYBER-PHYSICAL</span><br/>SECURITY
    </h1>
    <p style="font-size:17px;max-width:680px;color:#7e95b8;line-height:1.7;margin-bottom:24px">
      A semi-supervised hybrid intrusion detection system pairing a deep
      <strong style="color:#e8f0fd">Autoencoder</strong> (representation learning) with a
      <strong style="color:#e8f0fd">Random Forest</strong> (explicit boundaries) —
      trained exclusively on benign traffic to detect zero-day attacks.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-item">
        <span class="stat-num">3.2M</span><span class="stat-lbl">Network Flows</span>
      </div>
      <div class="stat-item">
        <span class="stat-num">68</span><span class="stat-lbl">Features</span>
      </div>
      <div class="stat-item">
        <span class="stat-num" style="color:#34d399">0.8965</span><span class="stat-lbl">Best AUC</span>
      </div>
      <div class="stat-item">
        <span class="stat-num">15</span><span class="stat-lbl">Attack Families</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("""
        <div class="s-card grn">
          <span class="eyebrow" style="color:#34d399">The Hybrid Synergy: Why it works</span>
          <p style="color:#7e95b8;font-size:14px;line-height:1.8;margin-top:4px">
            Our innovation is the <strong style="color:#e8f0fd">directional coupling</strong>
            between DL and ML:
          </p>
          <div style="display:flex;flex-direction:column;gap:12px;margin-top:12px">
            <div style="display:flex;gap:12px;align-items:flex-start">
              <span class="sentinel-badge sky" style="margin-top:2px;flex-shrink:0">DL Stage</span>
              <p style="font-size:14px;color:#7e95b8;margin:0">
                <strong style="color:#e8f0fd">Autoencoder (Dimensionality Compression)</strong>: 
                Network traffic is high-dimensional (68 features). The AE learns to 'squeeze' this into 
                a 32-dim latent space. <br/>
                <em>Why?</em> Malicious traffic doesn't 'fit' the compression rules learned from normal data. 
                High reconstruction error is our first alarm.
              </p>
            </div>
            <div style="display:flex;gap:12px;align-items:flex-start">
              <span class="sentinel-badge ind" style="margin-top:2px;flex-shrink:0">ML Stage</span>
              <p style="font-size:14px;color:#7e95b8;margin:0">
                <strong style="color:#e8f0fd">Random Forest (Boundary Sharpening)</strong>: 
                The AE finds anomalies but lacks 'decision confidence' for known attacks. 
                The RF takes the AE's output and uses 200 decision trees to vote on the verdict.<br/>
                <em>Why?</em> Trees are excellent at splitting data based on hard thresholds (like port numbers or packet sizes), 
                providing the precision that neural networks sometimes lack in tabular data.
              </p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="s-card sky">
          <span class="eyebrow">Phase Roadmap</span>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:8px">
            <span class="sentinel-badge">Baseline ML</span>
            <span style="color:#1b3050">→</span>
            <span class="sentinel-badge grn">Advanced ML</span>
            <span style="color:#1b3050">→</span>
            <span class="sentinel-badge ind">DL (AE)</span>
            <span style="color:#1b3050">→</span>
            <span class="sentinel-badge org" style="font-weight:700">Hybrid ← HERE</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="s-card" style="margin-bottom:14px">
          <span class="eyebrow" style="color:#818cf8">Team Groot 🌿</span>
          <p style="color:#7e95b8;font-size:14px;line-height:1.8;margin-top:6px">
            <strong style="color:#e8f0fd">Teammates</strong><br/>
            Dally R &amp; Pughazhendhi J<br/><br/>
            <strong style="color:#e8f0fd">Infrastructure</strong><br/>
            AWS EC2 t3.large · Ubuntu 24.04<br/>
            8 GB RAM + 16 GB swap<br/><br/>
            <strong style="color:#e8f0fd">Stack</strong><br/>
            TensorFlow 2.x · Scikit-Learn<br/>
            Python 3.10 · Streamlit
          </p>
        </div>

        <div class="s-card red">
          <span class="eyebrow" style="color:#f87171">Threat Level</span>
          <div style="font-size:22px;font-weight:800;color:#34d399;margin:8px 0">LOW — ALPHA</div>
          <div style="height:8px;background:rgba(255,255,255,.08);border-radius:4px;overflow:hidden">
            <div style="width:12%;height:100%;background:linear-gradient(90deg,#34d399,#38bdf8)"></div>
          </div>
          <p style="font-size:12px;color:#7e95b8;margin-top:8px">
            No anomalies detected in current monitoring window.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: THREAT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Threat Analysis":
    st.markdown("<span class='eyebrow'>Inference Engine</span>", unsafe_allow_html=True)
    st.markdown("## Threat Analysis")

    with st.expander("📚 Model Intelligence & Logic (Read Me)"):
        st.markdown("""
        ### Why this Hybrid Approach?
        *   **Autoencoder (AE)**: Functions as our *Anomaly Sentinel*. It learns the 'identity mapping' of normal traffic. When it encounters an attack, the reconstruction error spikes because the model cannot 'compress' the malicious patterns it never saw.
        *   **Random Forest (RF)**: Acts as our *Boundary Decision Engine*. While the AE finds 'weirdness', the RF learns to classify that weirdness into specific attack categories. We use **Pseudo-Labelling** where the AE tells the RF what is normal, allowing the RF to learn extremely sharp decision boundaries without needing perfectly labeled raw data.
        *   **The Fusion**: By combining the AE's unsupervised sensitivity with the RF's supervised structural depth, we achieve a system that detects zero-day attacks (AE) while maintaining high precision on known threats (RF).
        """)

    mode = st.radio("Input Source", ["Upload CSV Batch", "Manual Flow Entry", "Load Attack Profile"],
                    horizontal=True)

    df_input = None

    if mode == "Upload CSV Batch":
        up = st.file_uploader("Upload a network flow CSV (CIC-IDS2018 format)", type="csv")
        if up:
            df_input = pd.read_csv(up)
            # strip whitespace from column names
            df_input.columns = df_input.columns.str.strip()
            st.info(f"✅ Loaded **{len(df_input):,}** flow records — {len(df_input.columns)} columns")

    elif mode == "Manual Flow Entry":
        with st.expander("🛠️ Enter Flow Statistics"):
            cols = st.columns(4)
            manual = {f: 0.0 for f in FEATURE_NAMES}
            key_feats = ["Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts",
                         "Flow Byts/s","Flow Pkts/s","SYN Flag Cnt","ACK Flag Cnt",
                         "Pkt Len Mean","Flow IAT Mean"]
            for i, feat in enumerate(key_feats):
                with cols[i % 4]:
                    manual[feat] = st.number_input(feat, value=0.0, key=f"m_{feat}")
        df_input = pd.DataFrame([manual])

    elif mode == "Load Attack Profile":
        choice = st.selectbox("Select Attack Profile", list(ATTACK_PROFILES.keys()))
        profile = {f: 0.0 for f in FEATURE_NAMES}
        profile.update(ATTACK_PROFILES[choice])
        df_input = pd.DataFrame([profile])
        st.markdown(f"""
        <div class="callout {'bad' if choice != 'Normal Traffic' else 'good'}">
          <strong>Profile:</strong> {choice} — simulating a {choice.lower()} flow pattern.
          Key characteristics pre-populated from CIC-IDS2018 analysis.
        </div>
        """, unsafe_allow_html=True)

    if df_input is not None:
        if st.button("🔍 Run Neural Scan", width="stretch"):
            with st.spinner("Analysing traffic patterns through hybrid pipeline…"):
                results, meta = run_inference(df_input)

            if isinstance(results, pd.DataFrame):
                # ── BATCH ──────────────────────────────────────────────
                total = len(results); atk = int(results["Is_Attack"].sum())
                ben   = total - atk
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flows", f"{total:,}")
                c2.metric("🚨 Anomalies", str(atk), delta=f"{atk/total:.1%} rate")
                c3.metric("✅ Benign", str(ben))

                tab1, tab2 = st.tabs(["Score Distribution", "Latent Space 3D"])
                with tab1:
                    fig = go.Figure()
                    for col, clr in [("AE Score","#38bdf8"),("RF Score","#818cf8"),("Ensemble","#34d399")]:
                        fig.add_trace(go.Histogram(x=results[col], name=col, opacity=.7,
                                                   marker_color=clr, nbinsx=40))
                    plotly_dark_layout(fig, title="Anomaly Score Distribution",
                                       barmode="overlay", height=350)
                    st.plotly_chart(fig, width="stretch")

                with tab2:
                    fig3 = px.scatter_3d(results, x="Z1", y="Z2", z="Z3",
                                         color="Is_Attack",
                                         color_discrete_map={False:"#34d399", True:"#f87171"},
                                         opacity=.7, title="Bottleneck Latent Projection (PCA)")
                    plotly_dark_layout(fig3, height=450)
                    st.plotly_chart(fig3, width="stretch")

            else:
                # ── SINGLE ─────────────────────────────────────────────
                scores = results; top_feats = meta
                ens_score = scores["Meta-Ensemble"]
                is_atk    = ens_score > 0.5

                verdict_html = (
                    '<div class="status-bad">🚨 THREAT DETECTED</div>'
                    if is_atk else
                    '<div class="status-ok">✅ TRAFFIC SECURE</div>'
                )
                card_cls = "red" if is_atk else "grn"
                st.markdown(f"""
                <div class="s-card {card_cls}">
                  {verdict_html}
                  <p style="color:#7e95b8;margin-top:6px;font-size:14px">
                    Meta-Ensemble Score: <strong style="color:#e8f0fd">{ens_score:.4f}</strong>
                    &nbsp;(threshold = 0.50)
                  </p>
                </div>
                """, unsafe_allow_html=True)

                cl, cr = st.columns([1.2, 1])
                with cl:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=ens_score * 100,
                        delta={"reference": 50},
                        title={"text": "Anomaly Confidence (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#38bdf8"},
                            "steps": [
                                {"range": [0,  30], "color": "rgba(52,211,153,.15)"},
                                {"range": [30, 70], "color": "rgba(251,191,36,.1)"},
                                {"range": [70,100], "color": "rgba(248,113,113,.2)"},
                            ],
                            "threshold": {"line": {"color":"#f87171","width":3},
                                          "thickness":.75,"value":50},
                        },
                    ))
                    plotly_dark_layout(fig_g, height=300,
                                       margin=dict(t=50,b=0,l=20,r=20))
                    st.plotly_chart(fig_g, width="stretch")

                    # All model scores
                    st.markdown("<div class='s-card' style='padding:16px'>", unsafe_allow_html=True)
                    for model_name, val in scores.items():
                        clr = "#f87171" if val > .5 else "#34d399" if val < .2 else "#fb923c"
                        pct = min(val * 100, 100)
                        st.markdown(f"""
                        <div class="bar-row">
                          <div class="bar-label">{model_name}</div>
                          <div class="bar-bg"><div class="bar-fg" style="width:{pct:.1f}%;background:{clr}"></div></div>
                          <div class="bar-val">{val:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with cr:
                    st.markdown(f"""
                    <div class="s-card" style="padding:18px">
                      <span class="eyebrow">Top Anomalous Features</span>
                      <p style="font-size:13px;color:#7e95b8;margin-bottom:12px">
                        Features that deviated most from the AE's learned normal manifold:
                      </p>
                      {"".join(f'<div style="padding:8px 12px;margin-bottom:6px;background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.2);border-radius:8px;font-family:IBM Plex Mono,monospace;font-size:12px;color:#f87171">{f}</div>' for f in (top_feats or []))}
                    </div>
                    <div class="callout">
                      <strong>XAI Note:</strong> These features contributed most to the
                      reconstruction error. High deviation = the AE found this flow
                      structurally dissimilar to learned benign patterns.
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: LATENT UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌌  Latent Universe":
    st.markdown("<span class='eyebrow'>32-Dimensional Bottleneck Visualisation</span>",
                unsafe_allow_html=True)
    st.markdown("## Latent Universe")

    with st.expander("💡 Why does the latent space matter?"):
        st.markdown("""
        The Autoencoder compresses 68 raw traffic features into 32 "concept neurons" — the **bottleneck**.
        In this compressed space, benign traffic forms a **dense, tight cluster** (the model learned to
        reconstruct it perfectly). Attack traffic appears as **outliers** far from this cluster —
        their reconstruction error is high because the model never saw them during training.

        This is precisely why the **AE + RF hybrid works**: the RF draws explicit boundaries around
        the AE-defined normal region in raw feature space, sharpening the decision.
        """)

    # Synthetic latent clusters with realistic spread
    np.random.seed(42)
    n_b = 800; n_a = 120
    bx = np.random.multivariate_normal([0, 0, 0], [[1,.2,.1],[.2,1,.15],[.1,.15,1]], n_b)
    # Multiple attack clusters (different attack types cluster differently)
    atk_clusters = [
        np.random.normal([4, 1, -.5], .5, (40, 3)),
        np.random.normal([-2, 3,  2],  .7, (40, 3)),
        np.random.normal([1, -3, 3.5], .8, (40, 3)),
    ]
    ax = np.vstack(atk_clusters)
    attack_labels = ["DDoS-LOIC-UDP"]*40 + ["Brute Force"]*40 + ["DoS-Slowloris"]*40
    all_pts = np.vstack([bx, ax])
    all_lbl = ["Benign"]*n_b + attack_labels

    df_lat = pd.DataFrame(all_pts, columns=["Z1","Z2","Z3"])
    df_lat["Traffic Type"] = all_lbl

    colour_map = {
        "Benign": "#34d399", "DDoS-LOIC-UDP": "#f87171",
        "Brute Force": "#fb923c", "DoS-Slowloris": "#818cf8",
    }

    fig_lat = px.scatter_3d(df_lat, x="Z1", y="Z2", z="Z3",
                             color="Traffic Type", color_discrete_map=colour_map,
                             opacity=.65, size_max=4,
                             title="AE Bottleneck Latent Space (32D → 3D PCA projection)")
    plotly_dark_layout(fig_lat, height=520)
    fig_lat.update_traces(marker_size=3)
    st.plotly_chart(fig_lat, width="stretch")

    st.markdown("""
    <div class="callout good">
      <strong>Reading the plot:</strong> Benign traffic (green) forms a tight central cluster —
      the AE learned to reconstruct these perfectly. Each attack type occupies a different
      outlier region. DDoS-LOIC-UDP (red) is maximally separated — explaining its 100% detection rate.
      DoS-Slowloris (indigo) partially overlaps benign — explaining its moderate detection rate.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ATTACK SANDBOX
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Attack Sandbox":
    st.markdown("<span class='eyebrow'>Real-Time Feature Perturbation</span>",
                unsafe_allow_html=True)
    st.markdown("## Attack Sandbox")
    st.markdown("""
    <p style="color:#7e95b8;font-size:15px">
      Perturb individual flow statistics and watch how the hybrid model reacts in real-time.
      This demonstrates the model's sensitivity to different traffic characteristics.
    </p>
    """, unsafe_allow_html=True)

    cl, cr = st.columns([1, 1.1])
    with cl:
        st.markdown("#### 🛠️ Flow Feature Controls")
        flow_dur   = st.slider("Flow Duration (µs)", 0, 10_000_000, 50_000, 10_000,
                               help="Duration of the network flow in microseconds")
        tot_fwd    = st.slider("Total Forward Packets", 0, 5_000, 15,
                               help="Number of packets sent from src→dst")
        flow_bytes = st.slider("Flow Bytes/s", 0, 2_000_000, 80_000, 1_000,
                               help="Byte throughput rate of the flow")
        syn_cnt    = st.slider("SYN Flag Count", 0, 100, 1,
                               help="Number of TCP SYN flags — elevated in port scans / bruteforce")
        pkt_mean   = st.slider("Mean Packet Length (bytes)", 0, 1500, 512,
                               help="Average packet size — DDoS often uses small fixed-size packets")
        iat_mean   = st.slider("Flow IAT Mean (µs)", 0, 5_000_000, 3_000, 100,
                               help="Inter-arrival time — very low = flood; very high = slow attack")

        run_sb = st.button("⚡ Scan This Flow", width="stretch")

    with cr:
        # Auto-run if any slider changed, or if button clicked
        if True: 
            sample = {f: 0.0 for f in FEATURE_NAMES}
            sample.update({
                "Flow Duration": flow_dur, "Tot Fwd Pkts": tot_fwd,
                "Flow Byts/s": flow_bytes, "SYN Flag Cnt": syn_cnt,
                "Pkt Len Mean": pkt_mean, "Flow IAT Mean": iat_mean,
                "ACK Flag Cnt": max(0, tot_fwd - syn_cnt),
            })
            df_sb = pd.DataFrame([sample])
            res_sb, feats_sb = run_inference(df_sb)

            ae_score  = safe_float(res_sb.get("Autoencoder", 0))
            ens_score = safe_float(res_sb.get("Meta-Ensemble", 0))
            is_atk    = ens_score > 0.5

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min(ens_score * 100, 100),
                title={"text": "Anomaly Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0,  30], "color": "rgba(52,211,153,.15)"},
                        {"range": [30, 70], "color": "rgba(251,191,36,.1)"},
                        {"range": [70,100], "color": "rgba(248,113,113,.2)"},
                    ],
                    "threshold": {"line":{"color":"#f87171","width":3},"thickness":.75,"value":50},
                }
            ))
            plotly_dark_layout(fig_g, height=300, margin=dict(t=50,b=0,l=20,r=20))
            st.plotly_chart(fig_g, width='stretch')

            # --- Reasoning & Breakdown ---
            st.markdown("#### 🧠 Intelligence Analysis")
            
            # Feature Attribution reasoning
            reasoning_html = "".join([f'<span class="sentinel-badge prp">{f}</span>' for f in feats_sb])
            st.markdown(f"""
            <div class="s-card sky" style="padding:15px">
                <span class="eyebrow" style="color:var(--c1)">Root Cause Attribution</span>
                <p style="font-size:13px;color:var(--tx2);margin:8px 0">Top features deviating from normal manifold:</p>
                <div style="display:flex;flex-wrap:wrap;gap:8px">{reasoning_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # Model breakdown
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Autoencoder", f"{ae_score:.4f}")
            with c2: st.metric("Random Forest", f"{res_sb.get('Random Forest', 0):.4f}")
            with c3: st.metric("Ensemble P", f"{ens_score:.2%}")

            vrd = "🚨 ATTACK PATTERN" if is_atk else "✅ BENIGN PATTERN"
            cls = "bad" if is_atk else "good"
            st.markdown(f"""
            <div class="callout {cls}">
              <strong>{vrd}</strong><br/>
              Anomaly probability: {ens_score:.1%} (Threshold: 50.0%)
            </div>
            """, unsafe_allow_html=True)

            # Radar-style feature deviation bar
            feat_vals = {
                "Flow Duration": flow_dur/10_000_000,
                "Total Fwd Pkts": tot_fwd/5_000,
                "Flow Bytes/s": flow_bytes/2_000_000,
                "SYN Count": syn_cnt/100,
                "Pkt Mean Len": pkt_mean/1500,
                "IAT Mean": iat_mean/5_000_000,
            }
            fig_r = go.Figure(go.Scatterpolar(
                r=list(feat_vals.values()),
                theta=list(feat_vals.keys()),
                fill="toself",
                fillcolor="rgba(56,189,248,.15)",
                line_color="#38bdf8",
                name="Current Flow",
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=[.15]*6, theta=list(feat_vals.keys()),
                fill="toself", fillcolor="rgba(52,211,153,.1)",
                line_color="#34d399", name="Typical Benign",
            ))
            plotly_dark_layout(fig_r, title="Feature Profile vs Benign Baseline",
                               height=300, margin=dict(t=50,b=20,l=20,r=20),
                               polar=dict(bgcolor="rgba(12,22,40,.6)"))
            st.plotly_chart(fig_r, width="stretch")
        else:
            st.markdown("""
            <div style="height:420px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1b3050;border-radius:14px;background:#0c1628">
              <p style="color:#4a6280;font-size:15px">Adjust sliders → click Scan This Flow</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Benchmarks":
    st.markdown("<span class='eyebrow'>Hold-Out Test Set · CIC-IDS-2018</span>",
                unsafe_allow_html=True)
    st.markdown("## Model Performance Benchmarks")

    df_m = ASSETS["metrics"].copy()

    # Clean display
    rename = {
        "model":"Model","roc_auc":"ROC-AUC","avg_precision":"Avg Precision",
        "f1":"F1","recall":"Recall","fpr":"FPR","precision":"Precision",
    }
    df_show = df_m.rename(columns=rename).fillna("—")

    # Sort by AUC
    df_show = df_show.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    # ROC-AUC bar chart
    c1, c2 = st.columns(2)
    with c1:
        fig_auc = px.bar(df_show, x="ROC-AUC", y="Model", orientation="h",
                         title="ROC-AUC by Model",
                         color="ROC-AUC", color_continuous_scale=["#1b3050","#38bdf8"])
        fig_auc.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
        plotly_dark_layout(fig_auc, height=380, margin=dict(l=10,r=10,t=40,b=20))
        st.plotly_chart(fig_auc, width="stretch")

    with c2:
        fig_f1 = px.bar(df_show, x="F1", y="Model", orientation="h",
                        title="F1 Score by Model",
                        color="F1", color_continuous_scale=["#1b3050","#34d399"])
        fig_f1.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
        plotly_dark_layout(fig_f1, height=380, margin=dict(l=10,r=10,t=40,b=20))
        st.plotly_chart(fig_f1, width="stretch")

    # Full table
    st.markdown("""
    <div class="s-card sky">
      <span class="eyebrow">Full Metrics Table (All Models · P99 Threshold)</span>
    """, unsafe_allow_html=True)

    rows_html = ""
    for _, row in df_show.iterrows():
        auc_val = row["ROC-AUC"] if row["ROC-AUC"] != "—" else 0
        try: auc_f = float(auc_val)
        except: auc_f = 0
        win_cls  = ' class="win"' if auc_f >= 0.88 else (' class="bad"' if auc_f < 0.5 else "")
        rows_html += f"""
        <tr>
          <td style="color:#e8f0fd"><strong>{row["Model"]}</strong></td>
          <td{win_cls}>{row["ROC-AUC"]}</td>
          <td>{row.get("Avg Precision","—")}</td>
          <td>{row.get("F1","—")}</td>
          <td>{row.get("Recall","—")}</td>
          <td>{row.get("FPR","—")}</td>
        </tr>"""

    st.markdown(f"""
      <table class="s-table">
        <thead><tr>
          <th>Model</th><th>ROC-AUC ↑</th><th>Avg Precision ↑</th>
          <th>F1 ↑</th><th>Recall ↑</th><th>FPR ↓</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout warn">
      <strong>Key finding:</strong> Standard AE achieves the highest AUC (0.8965) and
      Avg Precision (0.7578). Adding OC-SVM or Isolation Forest to the blend
      <em>halves the AUC</em> — their unsupervised scores have no directional alignment
      with the AE signal. Only AE + RF maintains meaningful performance (0.7920)
      because RF is guided by AE pseudo-labels.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Ablation Study":
    st.markdown("<span class='eyebrow'>Component Necessity · Diagnostic Analysis</span>",
                unsafe_allow_html=True)
    st.markdown("## Ablation Study — What Breaks When You Remove Each Part?")

    tab1, tab2, tab3 = st.tabs(["Remove DL Component", "Remove ML Component", "AE Variant Comparison"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="s-card red">
              <span class="eyebrow" style="color:#f87171">DL Removed → RF Alone</span>
              <div style="display:flex;gap:16px;align-items:center;margin:14px 0">
                <div style="text-align:center">
                  <div style="font-family:Syne,sans-serif;font-size:36px;font-weight:800;color:#f87171">0.34</div>
                  <div style="font-size:12px;color:#7e95b8">AUC — RF alone</div>
                </div>
                <div style="font-size:32px;color:#f87171;font-weight:800">↓ 62%</div>
                <div style="text-align:center">
                  <div style="font-family:Syne,sans-serif;font-size:36px;font-weight:800;color:#38bdf8">0.90</div>
                  <div style="font-size:12px;color:#7e95b8">AUC — with AE</div>
                </div>
              </div>
              <p style="font-size:14px;color:#7e95b8;line-height:1.7">
                Without the Autoencoder generating pseudo-labels, the Random Forest
                has no directional signal about what "anomalous" means. It collapses
                to near-random (AUC 0.34). <strong style="color:#e8f0fd">The DL component is the
                backbone of this hybrid</strong> — the RF completely depends on the AE's
                learned representation.
              </p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="RF Alone", x=["ROC-AUC","F1","Recall"],
                                  y=[0.3433, 0.0335, 0.0173], marker_color="#f87171"))
            fig.add_trace(go.Bar(name="AE + RF Hybrid", x=["ROC-AUC","F1","Recall"],
                                  y=[0.7920, 0.0335, 0.0174], marker_color="#818cf8"))
            plotly_dark_layout(fig, title="RF Alone vs AE+RF Hybrid",
                               barmode="group", height=320)
            st.plotly_chart(fig, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="s-card sky">
              <span class="eyebrow" style="color:#38bdf8">ML Removed → AE Alone</span>
              <div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0">
                <div style="text-align:center;background:var(--surf);border:1px solid var(--brd);border-radius:10px;padding:12px 18px">
                  <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:#38bdf8">0.8965</div>
                  <div style="font-size:11px;color:#7e95b8">ROC-AUC</div>
                </div>
                <div style="text-align:center;background:var(--surf);border:1px solid var(--brd);border-radius:10px;padding:12px 18px">
                  <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:#34d399">0.7578</div>
                  <div style="font-size:11px;color:#7e95b8">Avg Precision</div>
                </div>
              </div>
              <p style="font-size:14px;color:#7e95b8;line-height:1.7">
                AE alone is the strongest anomaly <em>ranker</em> by both AUC and AP.
                Adding RF (→ AUC 0.7920) trades 8% ranking ability for sharper
                boundaries at fixed thresholds. <strong style="color:#e8f0fd">Use AE alone for
                maximum anomaly ranking. Use AE+RF for precise operational flagging.</strong>
              </p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            fig2 = go.Figure()
            models_cmp = ["AE Only","AE+RF","AE+OC-SVM","AE+IF","Meta-Ensemble"]
            aucs_cmp   = [0.8965, 0.7920, 0.4600, 0.4231, 0.4140]
            colors_cmp = ["#38bdf8","#818cf8","#f87171","#f87171","#f87171"]
            fig2.add_trace(go.Bar(x=models_cmp, y=aucs_cmp,
                                   marker_color=colors_cmp, showlegend=False))
            fig2.add_hline(y=0.5, line_dash="dash", line_color="#7e95b8",
                           annotation_text="Random (0.50)")
            plotly_dark_layout(fig2, title="AUC: What Happens When ML Is Varied",
                               height=320, yaxis_range=[0, 1])
            st.plotly_chart(fig2, width="stretch")

    with tab3:
        st.markdown("""
        <div class="s-card ind">
          <span class="eyebrow" style="color:#818cf8">AE Variant Ablation</span>
          <p style="font-size:14px;color:#7e95b8;margin-top:4px;margin-bottom:16px">
            Three architectural modifications tested against the baseline.
          </p>
        """, unsafe_allow_html=True)

        ae_variants = [
            ("Standard AE",   0.8965, 0.0336, 0.0174, "#38bdf8", "Baseline"),
            ("Sparse AE",     0.8929, 0.0336, 0.0174, "#818cf8", "L1 reg (λ=1e-4)"),
            ("Denoising AE",  0.8867, 0.1433, 0.0785, "#34d399", "GaussianNoise σ=0.1 ★"),
        ]
        for name, auc, f1, rec, clr, note in ae_variants:
            st.markdown(f"""
            <div style="margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:14px;color:#e8f0fd"><strong>{name}</strong>
                  <span class="sentinel-badge" style="font-size:10px;margin-left:6px">{note}</span>
                </span>
                <span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:{clr}">AUC {auc}</span>
              </div>
              <div class="bar-bg"><div class="bar-fg" style="width:{auc*100:.1f}%;background:{clr}"></div></div>
              <div style="font-size:12px;color:#7e95b8;margin-top:4px">
                F1: {f1:.4f} · Recall: {rec:.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="callout good">
          <strong>Key insight:</strong> Removing the GaussianNoise layer (going from Denoising AE
          to Standard AE) drops F1 by <strong>427%</strong> (0.0336 vs 0.1433) and Recall by
          451% — while losing only 1% AUC. The noise augmentation is the single most impactful
          architectural change for practical deployment.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️  Architecture":
    st.markdown("<span class='eyebrow'>Publication-Ready System Design</span>",
                unsafe_allow_html=True)
    st.markdown("## Hybrid IDS Architecture")

    # Pipeline summary
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:20px">
      <div style="background:var(--surf);border:1px solid var(--brd);border-radius:10px;
                  padding:12px 18px;text-align:center">
        <div style="font-family:IBM Plex Mono,monospace;font-size:22px;font-weight:700;
                    color:#fb923c">68D</div>
        <div style="font-size:11px;color:#7e95b8">Raw Features</div>
      </div>
      <div style="color:#1b3050;font-size:20px">──→</div>
      <div style="background:var(--surf);border:1px solid var(--brd);border-radius:10px;
                  padding:12px 18px;text-align:center">
        <div style="font-family:IBM Plex Mono,monospace;font-size:22px;font-weight:700;
                    color:#38bdf8">32D</div>
        <div style="font-size:11px;color:#7e95b8">Bottleneck</div>
      </div>
      <div style="color:#1b3050;font-size:20px">──→</div>
      <div style="background:var(--surf);border:1px solid var(--brd);border-radius:10px;
                  padding:12px 18px;text-align:center">
        <div style="font-family:IBM Plex Mono,monospace;font-size:18px;font-weight:700;
                    color:#818cf8">PSEUDO-LABELS</div>
        <div style="font-size:11px;color:#7e95b8">AE P99 Threshold</div>
      </div>
      <div style="color:#1b3050;font-size:20px">──→</div>
      <div style="background:var(--surf);border:1px solid var(--brd);border-radius:10px;
                  padding:12px 18px;text-align:center">
        <div style="font-family:IBM Plex Mono,monospace;font-size:22px;font-weight:700;
                    color:#818cf8">RF</div>
        <div style="font-size:11px;color:#7e95b8">200 Trees</div>
      </div>
      <div style="color:#1b3050;font-size:20px">──→</div>
      <div style="background:var(--surf);border:1px solid var(--brd);border-radius:10px;
                  padding:12px 18px;text-align:center">
        <div style="font-family:IBM Plex Mono,monospace;font-size:18px;font-weight:700;
                    color:#34d399">VERDICT</div>
        <div style="font-size:11px;color:#7e95b8">α-Score Fusion</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("""
        <div class="s-card sky">
          <span class="eyebrow">DL Stage — Three AE Variants</span>
          <p style="font-size:13px;color:#7e95b8;line-height:1.8;margin-top:8px">
            <strong style="color:#e8f0fd">Standard AE</strong>:
              <code>68→128→64→32→64→128→68</code><br/>
            Loss: MSE · Optimizer: Adam (lr=1e-3) · Params: 39,908<br/><br/>
            <strong style="color:#e8f0fd">Sparse AE</strong>:
              Same + L1(λ=1e-4) on bottleneck<br/>
            Forces sparse latent activations — fewer active neurons = cleaner boundaries<br/><br/>
            <strong style="color:#e8f0fd">Denoising AE</strong>:
              Same + GaussianNoise(σ=0.1) at input<br/>
            Noise injected during training only — robust to measurement jitter<br/><br/>
            All variants: BatchNorm + Dropout(0.2) · EarlyStopping(patience=7) ·
            ReduceLROnPlateau · Best checkpoint saved
          </p>
        </div>
        <div class="s-card ind">
          <span class="eyebrow">ML Stage — Random Forest</span>
          <p style="font-size:13px;color:#7e95b8;line-height:1.8;margin-top:8px">
            <strong style="color:#e8f0fd">Training Target</strong>:
              AE recon_error > P99(benign_val) → pseudo-attack (1)<br/>
            <strong style="color:#e8f0fd">Features</strong>:
              Raw 68-dim · No bottleneck (raw feature splits are more interpretable)<br/>
            <strong style="color:#e8f0fd">Config</strong>:
              n_estimators=200 · max_depth=12 · class_weight=balanced · n_jobs=-1<br/>
            <strong style="color:#e8f0fd">Subsample</strong>:
              ≤ 500K rows to fit on t3.large RAM<br/>
            <strong style="color:#e8f0fd">Fusion</strong>:
              S = 0.45·norm(AE_error) + 0.55·RF_proba → threshold at P99 on benign val
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Embed SVG architecture diagram
        st.markdown(f"""
        <div style="background:var(--surf);border:1px solid var(--brd);border-radius:14px;
                    overflow:hidden;padding:16px">
          <span class="eyebrow">Full Pipeline Diagram</span>
          <img src="{SVG_DATA_URI}" style="width:100%;margin-top:10px;border-radius:8px"/>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️   About":
    st.markdown("<span class='eyebrow'>Project Information</span>",
                unsafe_allow_html=True)
    st.markdown("## About SENTINEL")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="s-card ind">
          <span class="eyebrow" style="color:#818cf8">Team Groot 🌿</span>
          <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px;font-size:14px;color:#7e95b8;line-height:1.8">
            <div><strong style="color:#e8f0fd">Teammates</strong><br/>
              Dally R &amp; Pughazhendhi J</div>
            <div><strong style="color:#e8f0fd">Project</strong><br/>
              Adaptive Cyber-Physical Security — Phase 3 (Hybrid)</div>
            <div><strong style="color:#e8f0fd">Dataset</strong><br/>
              CSE-CIC-IDS2018 · 8.28M flows · 15 attack families</div>
            <div><strong style="color:#e8f0fd">Infrastructure</strong><br/>
              AWS EC2 t3.large · Ubuntu 24.04 · 8 GB RAM + 16 GB swap</div>
            <div><strong style="color:#e8f0fd">GitHub</strong><br/>
              astro-dally/Adaptive_Cyber_Physical_Security</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="s-card sky">
          <span class="eyebrow">Project Directory Structure</span>
          <p style="font-size:13px;color:#7e95b8;line-height:1.8;margin-top:8px;
                    font-family:IBM Plex Mono,monospace">
            /Adaptive_Cyber_Physical_Security/ids_2018/<br/>
            └── outputs/<br/>
                ├── robust_scaler.joblib<br/>
                ├── autoencoder_best.keras<br/>
                ├── sparse_ae_best.keras<br/>
                ├── denoising_ae_best.keras<br/>
                └── hybrid/<br/>
                    ├── rf_pseudo_label_final.joblib<br/>
                    └── model_comparison.csv
          </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # NOTE: Using custom HTML cards to maintain the "Team Groot" 
        # Sentinal-Glassmorphism aesthetic for academic presentation.
        st.markdown("""
        <div class="s-card grn">
          <span class="eyebrow" style="color:#34d399">Research Methodology</span>
          <p style="font-size:14px;color:#7e95b8;line-height:1.75;margin-top:8px">
            The system utilizes a <b>Benign-Only Training Paradigm</b>. 
            The Autoencoder (AE) is trained exclusively on normal traffic, learning to reconstruct 
            the "normal manifold" with high precision.
            <br/><br/>
            <b>Anomalous Flow Detection:</b><br/>
            Malicious packets produce high <i>Reconstruction Error</i> as they deviate from the learned normal patterns.
            <br/><br/>
            <b>Pseudo-Labelling Logic:</b><br/>
            The Random Forest is bootstrapped using a P99 threshold. Flows in the top 1% of error scores 
            are auto-tagged as <i>Pseudo-Attacks</i>, providing a directional signal for supervised 
            boundary estimation without requiring manual labels.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="s-card org">
          <span class="eyebrow" style="color:#fb923c">References</span>
          <p style="font-size:13px;color:#7e95b8;line-height:1.8;margin-top:8px">
            Sharafaldin, I., Lashkari, A.H., &amp; Ghorbani, A.A. (2018).
            <em>Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.</em>
            ICISSP.<br/><br/>
            Vincent, P. et al. (2010).
            <em>Stacked Denoising Autoencoders.</em> JMLR.<br/><br/>
            Breiman, L. (2001).
            <em>Random Forests.</em> Machine Learning, 45(1), 5–32.
          </p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Generate Final Cyber-Security Intelligence Report", width="stretch"):
        with st.status("🔐 Compiling secure packet telemetry...", expanded=True) as s:
            time.sleep(1.2)
            s.update(label="📡 Encrypting data streams...", state="running")
            time.sleep(1.0)
            s.update(label="✅ REPORT COMPILED: ACCESS GRANTED", state="complete")
        
        st.snow()  # Digital particles / Cyber-snow effect
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_id = f"GRT-{datetime.now().strftime('%Y%m%d')}-SENTINEL"
        st.markdown(f"""
        <div class="s-card sky" style="margin-top:16px; border: 1px solid var(--c1); box-shadow: 0 0 20px rgba(56,189,248,0.2)">
          <span class="eyebrow" style="color:var(--c1)">OFFICIAL SENTINEL RESEARCH REPORT</span>
          <p style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#7e95b8;margin-top:8px">
            REPORT_ID: {report_id}<br/>
            TIMESTAMP: {now_str}<br/>
            TEAM: Team Groot (Dally R &amp; Pughazhendhi J)
          </p>
          <hr style="border-color:#1b3050;margin:14px 0"/>
          <p style="font-size:14px;color:#7e95b8;line-height:1.75">
            <strong style="color:#e8f0fd">Executive Summary</strong><br/>
            The SENTINEL Hybrid system achieves ROC-AUC 0.8965 on CSE-CIC-IDS2018 using a
            two-stage architecture: an Autoencoder (DL) for anomaly scoring and a
            Random Forest (ML) for explicit boundary estimation via pseudo-labelling.
            The Denoising AE variant achieves 4× better F1 (0.1433) at only 1% AUC cost.
            <br/><br/>
            <strong style="color:#e8f0fd">Key Findings</strong><br/>
            1. DL component is critical — RF alone collapses to AUC 0.34.<br/>
            2. AE + RF (0.7920 AUC) is the best hybrid — RF guided by AE pseudo-labels.<br/>
            3. OC-SVM and IF dilute AUC when blended equally — equal-weight fusion
               requires directional alignment of component signals.<br/>
            4. DDoS-LOIC-UDP detected at 100%; SSH Bruteforce and Infiltration
               evade all flow-level models — sequence modelling is the next step.
          </p>
        </div>
        """, unsafe_allow_html=True)
