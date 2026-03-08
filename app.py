import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from utils import *
from explainability import make_gradcam_heatmap, overlay_heatmap
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection AI",
    layout="wide"
)

st.title("🩺 Multimodal Breast Cancer Detection System")
st.markdown("Mammogram + Ultrasound + Density Detection + Ensemble AI")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
models = load_models()
meta_model = load_meta_model()

ultra_models = load_ultrasound_models()
ultra_meta = load_ultrasound_meta()

density_model = load_density_model()
density_labels = load_density_classes()

labels = load_classes()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Upload & Predict", "Model Comparison", "Evaluation Dashboard"]
)

# ===================================================
# PAGE 1 — UPLOAD & PREDICT
# ===================================================
if menu == "Upload & Predict":

    st.header("Upload Mammogram Image")

    mammogram_file = st.file_uploader(
        "Upload Mammogram",
        type=["jpg","png","jpeg"]
    )

    if mammogram_file:

        image = Image.open(mammogram_file).convert("RGB")
        st.image(image, caption="Uploaded Mammogram", width=300)

        img = preprocess_image(image)

        # ---------------------------------------------------
        # DENSITY DETECTION
        # ---------------------------------------------------
        density_index = predict_density(density_model, img)
        density = density_labels[density_index]

        st.subheader("Breast Density Detection")
        st.write(f"Detected Density: **{density}**")

        if density == "D":
            st.warning("Dense breast detected — Ultrasound is recommended.")
        else:
            st.info("Breast not dense — Ultrasound optional.")

        # ---------------------------------------------------
        # INDIVIDUAL MAMMOGRAM PREDICTIONS
        # ---------------------------------------------------
        st.subheader("Mammogram Model Predictions")

        mammogram_results = {}

        for name, model in models.items():

            pred = model.predict(img)
            prob = float(pred.squeeze())

            label = labels[1] if prob > 0.5 else labels[0]

            mammogram_results[name] = prob

            st.write(f"**{name}** → {label} ({prob*100:.2f}%)")

        st.divider()

        # ---------------------------------------------------
        # MAMMOGRAM ENSEMBLE
        # ---------------------------------------------------
        st.subheader("Mammogram Stacking Ensemble")

        mammo_prob = float(stacking_predict(models, meta_model, img)[0])
        mammo_label = labels[1] if mammo_prob > 0.5 else labels[0]

        st.success(f"Ensemble Prediction: {mammo_label}")
        st.write(f"Probability: {mammo_prob*100:.2f}%")

        # ---------------------------------------------------
        # MODEL COMPARISON CHART
        # ---------------------------------------------------
        st.subheader("Model Probability Comparison")

        fig = plt.figure()
        plt.bar(mammogram_results.keys(), mammogram_results.values())
        plt.xticks(rotation=45)
        plt.ylabel("Probability of Malignant")
        plt.ylim(0,1)
        st.pyplot(fig)

        # ---------------------------------------------------
        # GRAD CAM
        # ---------------------------------------------------
        st.subheader("Explainable AI (Grad-CAM)")

        selected_model_name = st.selectbox(
            "Select Model for Grad-CAM",
            list(models.keys())
        )

        selected_model = models[selected_model_name]

        try:

            heatmap = make_gradcam_heatmap(img, selected_model)

            image_array = np.array(image)

            superimposed_img = overlay_heatmap(heatmap, image_array)

            st.image(superimposed_img, caption="Grad-CAM Heatmap")

        except Exception as e:

            st.warning("Grad-CAM could not be generated for this model.")
            st.write(str(e))

        # ===================================================
        # ULTRASOUND SECTION (PLACED AT BOTTOM)
        # ===================================================
        st.divider()
        st.header("Ultrasound Analysis (Optional)")

        ultrasound_file = st.file_uploader(
            "Upload Ultrasound Image",
            type=["jpg","png","jpeg"]
        )

        if ultrasound_file:

            us_image = Image.open(ultrasound_file).convert("RGB")
            st.image(us_image, caption="Uploaded Ultrasound", width=300)

            us_img = preprocess_image(us_image)

            # ---------------------------------------------------
            # ULTRASOUND INDIVIDUAL MODELS
            # ---------------------------------------------------
            st.subheader("Ultrasound Model Predictions")

            ultrasound_results = {}

            for name, model in ultra_models.items():

                pred = model.predict(us_img)
                prob = float(pred.squeeze())

                label = labels[1] if prob > 0.5 else labels[0]

                ultrasound_results[name] = prob

                st.write(f"**{name}** → {label} ({prob*100:.2f}%)")

            # ---------------------------------------------------
            # ULTRASOUND ENSEMBLE
            # ---------------------------------------------------
            st.subheader("Ultrasound Stacking Ensemble")

            ultra_prob = float(ultrasound_stacking_predict(ultra_models, ultra_meta, us_img)[0])
            ultra_label = labels[1] if ultra_prob > 0.5 else labels[0]

            st.success(f"Ensemble Prediction: {ultra_label}")
            st.write(f"Probability: {ultra_prob*100:.2f}%")

            # ---------------------------------------------------
            # FUSION
            # ---------------------------------------------------
            st.subheader("Multimodal Fusion Prediction")

            fusion_prob = fusion_predict(mammo_prob, ultra_prob)
            fusion_label = labels[1] if fusion_prob > 0.5 else labels[0]

            st.success(f"Final Prediction: {fusion_label}")
            st.write(f"Probability: {fusion_prob*100:.2f}%")

# ===================================================
# PAGE 2 — MODEL COMPARISON
# ===================================================
elif menu == "Model Comparison":

    st.header("Model Comparison")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(image)

        results = {}

        for name, model in models.items():

            pred = model.predict(img)
            prob = float(pred.squeeze())

            results[name] = prob

        st.write(results)

        fig = plt.figure()

        plt.bar(results.keys(), results.values())
        plt.xticks(rotation=45)
        plt.ylabel("Probability of Malignant")
        plt.ylim(0,1)

        st.pyplot(fig)

# ===================================================
# PAGE 3 — EVALUATION DASHBOARD
# ===================================================
elif menu == "Evaluation Dashboard":

    st.header("Evaluation Dashboard (Demo)")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(image)

        y_true = 1
        y_scores = []

        for model in models.values():

            pred = model.predict(img)
            prob = float(pred.squeeze())

            y_scores.append(prob)

        y_scores = np.array(y_scores)
        y_preds = (y_scores > 0.5).astype(int)

        # ---------------- Confusion Matrix ----------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix([y_true]*len(y_preds), y_preds)

        fig_cm, ax = plt.subplots()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        st.pyplot(fig_cm)

        # ---------------- ROC Curve ----------------
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve([y_true]*len(y_scores), y_scores)
        roc_auc = auc(fpr, tpr)

        fig_roc = plt.figure()

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        st.pyplot(fig_roc)