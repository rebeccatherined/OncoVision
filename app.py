import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from utils import load_models, load_meta_model, preprocess_image, load_classes, stacking_predict
from explainability import make_gradcam_heatmap, overlay_heatmap
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection AI",
    layout="wide"
)

st.title("🩺 AI-Based Breast Cancer Detection System")
st.markdown("Deep Learning | Transfer Learning | Stacking Ensemble | Explainable AI")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
models = load_models()
meta_model = load_meta_model()
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

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Mammogram", width=300)

        img = preprocess_image(image)

        st.subheader("Individual Model Predictions")

        results = {}

        for name, model in models.items():
            pred = model.predict(img)
            #prob = float(pred)
            prob = float(pred.squeeze())
            label = labels[1] if prob > 0.5 else labels[0]

            results[name] = prob

            st.write(f"**{name}** → {label} ({prob*100:.2f}%)")

        st.divider()

        # -------- STACKING ENSEMBLE --------
        st.subheader("Stacking Ensemble Prediction")

        final_prob = stacking_predict(models, meta_model, img)
        #final_prob = float(final_prob)
        final_prob = float(final_prob[0])

        final_label = labels[1] if final_prob > 0.5 else labels[0]

        st.success(f"Ensemble Prediction: {final_label}")
        st.write(f"Ensemble Probability: {final_prob*100:.2f}%")

        # -------- BAR CHART --------
        st.subheader("Model Probability Comparison")

        fig = plt.figure()
        plt.bar(results.keys(), results.values())
        plt.xticks(rotation=45)
        plt.ylabel("Probability of Malignant")
        plt.ylim(0, 1)
        st.pyplot(fig)

        # -------- GRAD-CAM --------
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
# PAGE 2 — MODEL COMPARISON
# ===================================================
elif menu == "Model Comparison":

    st.header("Model Comparison on Uploaded Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(image)

        results = {}

        for name, model in models.items():
            pred = model.predict(img)
            #results[name] = float(pred)
            results[name] = float(pred.squeeze())

        st.write("Prediction Probabilities:")
        st.write(results)

        fig = plt.figure()
        plt.bar(results.keys(), results.values())
        plt.xticks(rotation=45)
        plt.ylabel("Probability of Malignant")
        plt.ylim(0, 1)
        st.pyplot(fig)


# ===================================================
# PAGE 3 — EVALUATION DASHBOARD (DEMO MODE)
# ===================================================
elif menu == "Evaluation Dashboard":

    st.header("Evaluation Metrics (Demo Mode)")
    st.info("Upload an image to visualize Confusion Matrix & ROC Curve")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(image)

        # Demo ground truth (for visualization)
        y_true = 1

        y_scores = []

        for model in models.values():
            pred = model.predict(img)
            #y_scores.append(float(pred))
            y_scores.append(float(pred.squeeze()))

        y_scores = np.array(y_scores)
        y_preds = (y_scores > 0.5).astype(int)

        # -------- CONFUSION MATRIX --------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix([y_true]*len(y_preds), y_preds)

        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig_cm)

        # -------- ROC CURVE --------
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve([y_true]*len(y_scores), y_scores)
        roc_auc = auc(fpr, tpr)

        fig_roc = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig_roc)