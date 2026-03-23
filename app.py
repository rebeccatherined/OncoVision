import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image

from utils import (
    load_models, load_meta_model,
    load_ultrasound_models, load_ultrasound_meta,
    load_density_model, load_density_meta, predict_density,
    load_classes,
    preprocess_image, stacking_predict,
    fusion_predict, interpret_risk
)
from explainability import make_gradcam_heatmap, overlay_heatmap

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastAI — Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2a3a 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
.main { background: #0d1117; }
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #4fc3f7;
    border-left: 3px solid #4fc3f7; padding-left: 12px; margin: 24px 0 14px;
}
.result-banner { border-radius: 12px; padding: 18px 24px; margin: 14px 0; font-size: 1rem; font-weight: 500; }
.result-malignant { background: linear-gradient(135deg,#3d1a1a,#5c1f1f); border:1px solid #c62828; color:#ef9a9a; }
.result-benign    { background: linear-gradient(135deg,#1a2d1a,#1e3a1e); border:1px solid #2e7d32; color:#a5d6a7; }
.result-moderate  { background: linear-gradient(135deg,#2d2a1a,#3a351e); border:1px solid #f57f17; color:#ffe082; }
.density-badge { display:inline-block; padding:5px 14px; border-radius:20px; font-weight:600; font-size:0.95rem; }
.density-dense    { background:#b71c1c; color:#ef9a9a; }
.density-nondense { background:#1b5e20; color:#a5d6a7; }
.metric-tile { background:#131f2e; border:1px solid #1e3a5f; border-radius:10px; padding:12px; text-align:center; }
.metric-tile .metric-val   { font-size:1.5rem; font-weight:600; color:#4fc3f7; }
.metric-tile .metric-label { font-size:0.75rem; color:#7ba3c8; margin-top:2px; }
[data-testid="stFileUploader"] { background:#131f2e; border:1.5px dashed #1e3a5f; border-radius:12px; padding:10px; }
hr { border-color:#1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD ALL MODELS
# ─────────────────────────────────────────────────────────────────
with st.spinner("Loading AI models — please wait..."):
    models        = load_models()
    meta_model    = load_meta_model()
    ultra_models  = load_ultrasound_models()
    ultra_meta    = load_ultrasound_meta()
    density_model = load_density_model()
    density_meta  = load_density_meta()
    labels        = load_classes()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px;'>
        <div style='font-size:2.2rem'>🩺</div>
        <div style='font-size:1.15rem;font-weight:600;color:#4fc3f7;'>BreastAI</div>
        <div style='font-size:0.75rem;color:#5c8ab0;margin-top:4px;'>Multimodal Cancer Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    menu = st.selectbox("Navigation",
        ["🔬 Predict & Analyse", "📊 Model Comparison", "📈 Evaluation Dashboard", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#5c8ab0;line-height:1.8;'>
    <b style='color:#7ba3c8'>Mammogram (5 models)</b><br>
    · Custom ResNet CNN<br>· VGG16 Feature<br>· VGG16 FineTune<br>· DenseNet121<br>· MobileNetV2<br><br>
    <b style='color:#7ba3c8'>Ultrasound (4 models)</b><br>
    · Custom Attention CNN<br>· DenseNet121<br>· MobileNetV2<br>· InceptionV3<br><br>
    <b style='color:#7ba3c8'>Density (1 model)</b><br>
    · MobileNetV2 (Dense/Non-Dense)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def result_banner(prob, threshold, label_map):
    label = label_map[1] if prob > threshold else label_map[0]
    risk, color, icon = interpret_risk(prob, threshold)
    css = {"Low Risk":"result-benign","Moderate Risk":"result-moderate","High Risk":"result-malignant"}[risk]
    st.markdown(f"""
    <div class="result-banner {css}">
        {icon} &nbsp; <b>{label}</b> &nbsp;·&nbsp; {risk} &nbsp;·&nbsp; Probability: <b>{prob*100:.1f}%</b>
        &nbsp;·&nbsp; Threshold: {threshold:.2f}
    </div>""", unsafe_allow_html=True)


def plotly_bar(results, title):
    names  = list(results.keys())
    probs  = [v * 100 for v in results.values()]
    colors = ["#ef9a9a" if p > 50 else "#a5d6a7" for p in probs]
    fig = go.Figure(go.Bar(
        x=names, y=probs, marker_color=colors,
        text=[f"{p:.1f}%" for p in probs], textposition="outside"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#4fc3f7", size=13)),
        paper_bgcolor="#131f2e", plot_bgcolor="#131f2e",
        font=dict(color="#c8d8e8"),
        yaxis=dict(range=[0,120], title="Malignant Probability (%)", gridcolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        margin=dict(l=10,r=10,t=40,b=10), height=300
    )
    return fig


def render_individual_models(results, labels, title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    for name, prob in results.items():
        label = labels[1] if prob > 0.5 else labels[0]
        color = "#ef9a9a" if prob > 0.5 else "#a5d6a7"
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<span style="color:#90caf9;font-weight:500;">🤖 {name}</span>', unsafe_allow_html=True)
            st.progress(float(prob))
        with col2:
            st.markdown(f'<span style="color:{color};font-family:monospace;font-size:0.85rem;">{label}<br>{prob*100:.1f}%</span>', unsafe_allow_html=True)


def gauge_chart(prob, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': title, 'font': {'color':'#c8d8e8'}},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': "#ef9a9a" if prob > 0.5 else "#a5d6a7"},
            'steps': [
                {'range':[0,30],  'color':'#1b5e20'},
                {'range':[30,60], 'color':'#e65100'},
                {'range':[60,100],'color':'#b71c1c'},
            ],
            'threshold': {'line':{'color':'white','width':2},'value':50}
        },
        number={'font':{'color':'#4fc3f7'},'suffix':'%'}
    ))
    fig.update_layout(paper_bgcolor="#131f2e", font=dict(color="#c8d8e8"),
                      height=230, margin=dict(l=20,r=20,t=30,b=10))
    return fig


# ─────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT & ANALYSE
# ─────────────────────────────────────────────────────────────────
if menu == "🔬 Predict & Analyse":

    st.markdown("""
    <h1 style='color:#4fc3f7;margin-bottom:4px;'>🩺 Breast Cancer Detection</h1>
    <p style='color:#5c8ab0;font-size:0.88rem;'>Upload a mammogram for AI-powered analysis from 5 models + ensemble. Ultrasound available for dense breasts.</p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Step 1 — Upload Mammogram</div>', unsafe_allow_html=True)
    mammo_file = st.file_uploader("", type=["jpg","png","jpeg"], key="mammo", label_visibility="collapsed")

    if mammo_file:
        pil_img = Image.open(mammo_file).convert("RGB")
        img_224 = preprocess_image(pil_img, (224,224))

        col_img, col_density = st.columns([1, 2])

        with col_img:
            st.image(pil_img, caption="Uploaded Mammogram", width=260)

        with col_density:
            # ── Density Detection ──
            st.markdown('<div class="section-header">Breast Density</div>', unsafe_allow_html=True)
            is_dense, prob_dense, density_label = predict_density(density_model, img_224, density_meta)
            badge_cls = "density-dense" if is_dense else "density-nondense"
            msg = "⚠️ Dense breast — Ultrasound recommended!" if is_dense else "✅ Non-dense breast — Ultrasound optional."
            st.markdown(f"""
            <div>
                <span class="density-badge {badge_cls}">{density_label}</span>
                &nbsp;<span style='color:#7ba3c8;font-size:0.82rem;'>Confidence: {prob_dense*100:.1f}%</span>
            </div>
            <p style='color:#aac4de;font-size:0.88rem;margin-top:8px;'>{msg}</p>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Individual Mammogram Model Predictions ──
        mammo_results = {}
        with st.spinner("Running 5 mammogram models..."):
            for name, model in models.items():
                mammo_results[name] = float(model.predict(img_224, verbose=0).squeeze())

        col_l, col_r = st.columns(2)
        with col_l:
            render_individual_models(mammo_results, labels, "Mammogram — Individual Models")
        with col_r:
            st.plotly_chart(plotly_bar(mammo_results, "Malignancy Probability by Model"), use_container_width=True)

        # ── Mammogram Ensemble ──
        st.markdown('<div class="section-header">Mammogram Stacking Ensemble</div>', unsafe_allow_html=True)
        with st.spinner("Computing ensemble..."):
            mammo_prob, mammo_thresh = stacking_predict(models, meta_model, img_224, pil_img)
        result_banner(mammo_prob, mammo_thresh, labels)

        # ── Grad-CAM ──
        with st.expander("🔍 Explainable AI — Grad-CAM Heatmap", expanded=False):
            gradcam_model_name = st.selectbox(
                "Select model",
                [k for k in models.keys() if "VGG" in k or "Dense" in k or "Mobile" in k]
            )
            if st.button("Generate Heatmap"):
                with st.spinner("Generating Grad-CAM..."):
                    try:
                        heatmap = make_gradcam_heatmap(img_224, models[gradcam_model_name])
                        img_arr = np.array(pil_img.resize((224,224)))
                        superimposed = overlay_heatmap(heatmap, img_arr)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(pil_img.resize((224,224)), caption="Original", width=220)
                        with c2:
                            st.image(superimposed, caption=f"Grad-CAM ({gradcam_model_name})", width=220)
                        st.info("🔴 Red = regions model focused on most.")
                    except Exception as e:
                        st.warning(f"Grad-CAM unavailable for this model: {e}")

        # ── Ultrasound Section ──
        st.markdown("---")
        us_header = "Step 2 — Ultrasound Analysis (Recommended ⚠️)" if is_dense else "Step 2 — Ultrasound Analysis (Optional)"
        st.markdown(f'<div class="section-header">{us_header}</div>', unsafe_allow_html=True)

        if is_dense:
            st.warning("Dense breast detected. Ultrasound provides better detection for dense tissue.")
        else:
            st.info("Breast is non-dense. You may still upload an ultrasound for a more comprehensive analysis.")

        us_file = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"], key="us")

        if us_file:
            us_pil = Image.open(us_file).convert("RGB")
            us_img = preprocess_image(us_pil, (224,224))

            col_us, _ = st.columns([1, 2])
            with col_us:
                st.image(us_pil, caption="Uploaded Ultrasound", width=260)

            # Individual ultrasound models
            us_results = {}
            with st.spinner("Running 4 ultrasound models..."):
                for name, model in ultra_models.items():
                    if name == "InceptionV3":
                        inp299 = preprocess_image(us_pil, (299,299))
                        us_results[name] = float(model.predict(inp299, verbose=0).squeeze())
                    else:
                        us_results[name] = float(model.predict(us_img, verbose=0).squeeze())

            col_ul, col_ur = st.columns(2)
            with col_ul:
                render_individual_models(us_results, labels, "Ultrasound — Individual Models")
            with col_ur:
                st.plotly_chart(plotly_bar(us_results, "Ultrasound Malignancy Probability"), use_container_width=True)

            # Ultrasound ensemble
            st.markdown('<div class="section-header">Ultrasound Stacking Ensemble</div>', unsafe_allow_html=True)
            with st.spinner("Computing ultrasound ensemble..."):
                ultra_prob, ultra_thresh = stacking_predict(ultra_models, ultra_meta, us_img, us_pil)
            result_banner(ultra_prob, ultra_thresh, labels)

            # Multimodal Fusion
            st.markdown('<div class="section-header">🔀 Multimodal Fusion — Final Prediction</div>', unsafe_allow_html=True)
            fusion_prob = fusion_predict(mammo_prob, ultra_prob)

            c1, c2, c3 = st.columns(3)
            for col, label_text, val in [(c1,"Mammogram Ensemble",mammo_prob),(c2,"Ultrasound Ensemble",ultra_prob),(c3,"Fusion (60%+40%)",fusion_prob)]:
                with col:
                    st.markdown(f"""
                    <div style='background:#131f2e;border:1px solid #1e3a5f;border-radius:10px;padding:14px;text-align:center;'>
                        <div style='font-size:0.78rem;color:#7ba3c8;'>{label_text}</div>
                        <div style='font-size:1.6rem;font-weight:700;color:#4fc3f7;'>{val*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("**🎯 Final Fused Prediction:**")
            result_banner(fusion_prob, 0.5, labels)
            st.plotly_chart(gauge_chart(fusion_prob, "Malignancy Risk (%)"), use_container_width=True)

        else:
            st.plotly_chart(gauge_chart(mammo_prob, "Mammogram Risk (%)"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 2 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────
elif menu == "📊 Model Comparison":
    st.markdown("<h1 style='color:#4fc3f7;'>📊 Model Comparison</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Mammogram Models", "Ultrasound Models"])

    with tab1:
        f = st.file_uploader("Upload Mammogram", type=["jpg","png","jpeg"], key="cmp_m")
        if f:
            pil = Image.open(f).convert("RGB")
            img = preprocess_image(pil)
            results = {name: float(m.predict(img, verbose=0).squeeze()) for name, m in models.items()}
            col1, col2 = st.columns([1,2])
            with col1: st.image(pil, width=240)
            with col2: st.plotly_chart(plotly_bar(results, "Mammogram — All Models"), use_container_width=True)
            import pandas as pd
            st.dataframe(pd.DataFrame({
                "Model": list(results.keys()),
                "Probability (%)": [f"{v*100:.2f}" for v in results.values()],
                "Prediction": [labels[1] if v > 0.5 else labels[0] for v in results.values()]
            }), use_container_width=True, hide_index=True)

    with tab2:
        f2 = st.file_uploader("Upload Ultrasound", type=["jpg","png","jpeg"], key="cmp_u")
        if f2:
            pil2 = Image.open(f2).convert("RGB")
            img2 = preprocess_image(pil2)
            us_res = {}
            for name, m in ultra_models.items():
                if name == "InceptionV3":
                    us_res[name] = float(m.predict(preprocess_image(pil2,(299,299)), verbose=0).squeeze())
                else:
                    us_res[name] = float(m.predict(img2, verbose=0).squeeze())
            col1, col2 = st.columns([1,2])
            with col1: st.image(pil2, width=240)
            with col2: st.plotly_chart(plotly_bar(us_res, "Ultrasound — All Models"), use_container_width=True)
            import pandas as pd
            st.dataframe(pd.DataFrame({
                "Model": list(us_res.keys()),
                "Probability (%)": [f"{v*100:.2f}" for v in us_res.values()],
                "Prediction": [labels[1] if v > 0.5 else labels[0] for v in us_res.values()]
            }), use_container_width=True, hide_index=True)


# # ─────────────────────────────────────────────────────────────────
# # PAGE 3 — EVALUATION DASHBOARD
# # ─────────────────────────────────────────────────────────────────
# elif menu == "📈 Evaluation Dashboard":
#     st.markdown("<h1 style='color:#4fc3f7;'>📈 Evaluation Dashboard</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='color:#5c8ab0;'>Upload an image with its true label to see per-model metrics and agreement.</p>", unsafe_allow_html=True)

#     col_f, col_l = st.columns([2,1])
#     with col_f:
#         ev_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"], key="eval")
#     with col_l:
#         true_label = st.radio("True Label", ["Benign","Malignant"], horizontal=True)

#     if ev_file:
#         pil_ev = Image.open(ev_file).convert("RGB")
#         img_ev = preprocess_image(pil_ev)
#         y_true = 1 if true_label == "Malignant" else 0

#         scores = {name: float(m.predict(img_ev, verbose=0).squeeze()) for name, m in models.items()}
#         preds  = {k: (1 if v > 0.5 else 0) for k, v in scores.items()}

#         st.markdown('<div class="section-header">Per-Model Results</div>', unsafe_allow_html=True)
#         import pandas as pd
#         st.dataframe(pd.DataFrame({
#             "Model": list(scores.keys()),
#             "Prediction": [labels[1] if preds[k]==1 else labels[0] for k in scores],
#             "Correct": ["✅" if preds[k]==y_true else "❌" for k in scores],
#             "Probability": [f"{v*100:.1f}%" for v in scores.values()],
#         }), use_container_width=True, hide_index=True)

#         # Agreement pie
#         n_mal = sum(1 for v in preds.values() if v==1)
#         n_ben = len(preds) - n_mal
#         fig_pie = go.Figure(go.Pie(
#             labels=["Malignant Vote","Benign Vote"], values=[n_mal, n_ben],
#             hole=0.5, marker_colors=["#ef9a9a","#a5d6a7"]
#         ))
#         fig_pie.update_layout(paper_bgcolor="#131f2e", font=dict(color="#c8d8e8"),
#                                height=250, margin=dict(l=0,r=0,t=30,b=0))
#         st.plotly_chart(fig_pie, use_container_width=True)

#         # Ensemble result
#         st.markdown('<div class="section-header">Ensemble Result</div>', unsafe_allow_html=True)
#         ens_prob, ens_thresh = stacking_predict(models, meta_model, img_ev, pil_ev)
#         result_banner(ens_prob, ens_thresh, labels)
#         correct = "✅ Correct" if (ens_prob > ens_thresh) == bool(y_true) else "❌ Incorrect"
#         st.markdown(f"True Label: **{true_label}** | Ensemble: **{correct}**")


# ─────────────────────────────────────────────────────────────────
# PAGE 3 — EVALUATION DASHBOARD
# ─────────────────────────────────────────────────────────────────
elif menu == "📈 Evaluation Dashboard":
    st.markdown("<h1 style='color:#4fc3f7;'>📈 Evaluation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5c8ab0;'>Upload an image with its true label to see per-model metrics and agreement.</p>", unsafe_allow_html=True)

    eval_tab1, eval_tab2 = st.tabs(["Mammogram Models", "Ultrasound Models"])

    # ── MAMMOGRAM TAB ──────────────────────────────────────────
    with eval_tab1:
        col_f, col_l = st.columns([2, 1])
        with col_f:
            ev_file = st.file_uploader("Upload Mammogram Image", type=["jpg","png","jpeg"], key="eval_mammo")
        with col_l:
            true_label = st.radio("True Label", ["Benign","Malignant"], horizontal=True, key="lbl_mammo")

        if ev_file:
            pil_ev = Image.open(ev_file).convert("RGB")
            img_ev = preprocess_image(pil_ev)
            y_true = 1 if true_label == "Malignant" else 0

            scores = {name: float(m.predict(img_ev, verbose=0).squeeze()) for name, m in models.items()}
            preds  = {k: (1 if v > 0.5 else 0) for k, v in scores.items()}

            st.markdown('<div class="section-header">Per-Model Results</div>', unsafe_allow_html=True)
            import pandas as pd
            st.dataframe(pd.DataFrame({
                "Model":      list(scores.keys()),
                "Prediction": [labels[1] if preds[k]==1 else labels[0] for k in scores],
                "Correct":    ["✅" if preds[k]==y_true else "❌" for k in scores],
                "Probability":[f"{v*100:.1f}%" for v in scores.values()],
            }), use_container_width=True, hide_index=True)

            # Agreement pie
            n_mal = sum(1 for v in preds.values() if v==1)
            n_ben = len(preds) - n_mal
            fig_pie = go.Figure(go.Pie(
                labels=["Malignant Vote","Benign Vote"],
                values=[n_mal, n_ben],
                hole=0.5, marker_colors=["#ef9a9a","#a5d6a7"]
            ))
            fig_pie.update_layout(
                paper_bgcolor="#131f2e", font=dict(color="#c8d8e8"),
                height=250, margin=dict(l=0,r=0,t=30,b=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Probability bar chart
            st.plotly_chart(plotly_bar(scores, "Mammogram — Model Probabilities"), use_container_width=True)

            # Ensemble
            st.markdown('<div class="section-header">Ensemble Result</div>', unsafe_allow_html=True)
            ens_prob, ens_thresh = stacking_predict(models, meta_model, img_ev, pil_ev)
            result_banner(ens_prob, ens_thresh, labels)
            correct = "✅ Correct" if (ens_prob > ens_thresh) == bool(y_true) else "❌ Incorrect"
            st.markdown(f"True Label: **{true_label}** | Ensemble: **{correct}**")

    # ── ULTRASOUND TAB ─────────────────────────────────────────
    with eval_tab2:
        col_f2, col_l2 = st.columns([2, 1])
        with col_f2:
            ev_file_us = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"], key="eval_us")
        with col_l2:
            true_label_us = st.radio("True Label", ["Benign","Malignant"], horizontal=True, key="lbl_us")

        if ev_file_us:
            pil_ev_us = Image.open(ev_file_us).convert("RGB")
            img_ev_us = preprocess_image(pil_ev_us)
            y_true_us = 1 if true_label_us == "Malignant" else 0

            us_scores = {}
            for name, m in ultra_models.items():
                if name == "InceptionV3":
                    inp299 = preprocess_image(pil_ev_us, (299,299))
                    us_scores[name] = float(m.predict(inp299, verbose=0).squeeze())
                else:
                    us_scores[name] = float(m.predict(img_ev_us, verbose=0).squeeze())

            us_preds = {k: (1 if v > 0.5 else 0) for k, v in us_scores.items()}

            st.markdown('<div class="section-header">Per-Model Results</div>', unsafe_allow_html=True)
            import pandas as pd
            st.dataframe(pd.DataFrame({
                "Model":      list(us_scores.keys()),
                "Prediction": [labels[1] if us_preds[k]==1 else labels[0] for k in us_scores],
                "Correct":    ["✅" if us_preds[k]==y_true_us else "❌" for k in us_scores],
                "Probability":[f"{v*100:.1f}%" for v in us_scores.values()],
            }), use_container_width=True, hide_index=True)

            # Agreement pie
            n_mal_us = sum(1 for v in us_preds.values() if v==1)
            n_ben_us = len(us_preds) - n_mal_us
            fig_pie_us = go.Figure(go.Pie(
                labels=["Malignant Vote","Benign Vote"],
                values=[n_mal_us, n_ben_us],
                hole=0.5, marker_colors=["#ef9a9a","#a5d6a7"]
            ))
            fig_pie_us.update_layout(
                paper_bgcolor="#131f2e", font=dict(color="#c8d8e8"),
                height=250, margin=dict(l=0,r=0,t=30,b=0)
            )
            st.plotly_chart(fig_pie_us, use_container_width=True)

            # Probability bar chart
            st.plotly_chart(plotly_bar(us_scores, "Ultrasound — Model Probabilities"), use_container_width=True)

            # Ensemble
            st.markdown('<div class="section-header">Ultrasound Ensemble Result</div>', unsafe_allow_html=True)
            ens_prob_us, ens_thresh_us = stacking_predict(ultra_models, ultra_meta, img_ev_us, pil_ev_us)
            result_banner(ens_prob_us, ens_thresh_us, labels)
            correct_us = "✅ Correct" if (ens_prob_us > ens_thresh_us) == bool(y_true_us) else "❌ Incorrect"
            st.markdown(f"True Label: **{true_label_us}** | Ensemble: **{correct_us}**")


# ─────────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────────────────────────
elif menu == "ℹ️ About":
    st.markdown("<h1 style='color:#4fc3f7;'>ℹ️ About This System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#aac4de;line-height:1.9;font-size:0.92rem;'>
    <h3 style='color:#4fc3f7;'>Project Overview</h3>
    A multimodal AI system for breast cancer detection combining mammogram and ultrasound analysis
    using deep learning, transfer learning, stacking ensembles, and explainable AI (Grad-CAM).

    <h3 style='color:#4fc3f7;'>System Pipeline</h3>
    <ol>
        <li>Upload mammogram → 5 models generate individual predictions</li>
        <li>Stacking ensemble (Logistic Regression meta-learner) combines predictions</li>
        <li>MobileNetV2 classifies breast density (Dense vs Non-Dense)</li>
        <li>Dense breast → ultrasound recommended (4 models + ensemble)</li>
        <li>If ultrasound provided → weighted fusion (60% mammo + 40% ultrasound)</li>
        <li>Grad-CAM highlights regions the model focused on</li>
    </ol>

    <h3 style='color:#4fc3f7;'>Models</h3>
    <b>Mammogram:</b> Custom ResNet CNN · VGG16 Feature · VGG16 FineTune · DenseNet121 · MobileNetV2<br>
    <b>Ultrasound:</b> Custom Attention CNN · DenseNet121 · MobileNetV2 · InceptionV3<br>
    <b>Density:</b> MobileNetV2 (Binary: Dense / Non-Dense, AUC=0.949)<br>

    <h3 style='color:#4fc3f7;'>Datasets</h3>
    Mammogram: Custom clinical dataset · Ultrasound: BUSI (Kaggle) · Density: MIAS Mammography

    <h3 style='color:#4fc3f7;'>⚠️ Disclaimer</h3>
    For academic/research purposes only. Not a substitute for professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)