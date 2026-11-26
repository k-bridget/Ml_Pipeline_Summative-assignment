import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Webpage Phishing Detection",
    layout="wide",
)

API_URL = os.environ.get("PHISHIRIS_API_URL", "http://127.0.0.1:8000")

st.title("Webpage Phishing Detection")
st.caption("End-to-end computer vision pipeline: single-image prediction, feature insights, bulk upload, and retraining.")

# Sidebar: Model Uptime & API Health
with st.sidebar:
    st.header("System Status & Uptime")
    st.text_input("API Base URL", value=API_URL, key="api_url_input")
    base_url = st.session_state["api_url_input"].rstrip("/")

    if st.button("Refresh Health Status"):
        pass

    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.ok:
            data = resp.json()
            st.success("API Online")
            st.metric("Model Loaded", "Yes" if data.get("model_loaded") else "No")
            st.metric("Uptime (sec)", data.get("uptime_seconds", 0))
        else:
            st.warning(f"API returned status {resp.status_code}")
    except Exception as exc:
        st.error(f"API unreachable: {exc}")

tab_predict, tab_viz, tab_retrain = st.tabs(
    ["Predict Screenshot", "Feature Visualizations ", "Bulk Upload & Retrain"]
)

# Tab 1: Single Datapoint Prediction
with tab_predict:
    st.subheader("Predict a Single Webpage Screenshot")
    uploaded_img = st.file_uploader(
        "Upload a screenshot (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        key="single_pred_uploader",
    )
    if uploaded_img is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_img, caption=uploaded_img.name, use_container_width=True)

        with col2:
            if st.button("Predict", type="primary"):
                uploaded_img.seek(0)
                files = {"file": (uploaded_img.name, uploaded_img.getvalue(), uploaded_img.type or "image/png")}
                try:
                    r = requests.post(f"{base_url}/predict", files=files, timeout=30)
                    if r.ok:
                        out = r.json()
                        pred = out.get("prediction", "Unknown")
                        conf = out.get("confidence", 0.0)
                        prob = out.get("phishing_probability", 0.0)
                        if pred == "Phishing":
                            st.error(f"Prediction: **PHISHING** (Confidence: {conf:.2%})")
                        else:
                            st.success(f"Prediction: **LEGITIMATE** (Confidence: {conf:.2%})")
                        st.write(f"**Phishing Probability:** `{prob:.4f}`")
                        if "features" in out:
                            st.write("**Extracted Image Features:**")
                            st.json(out["features"])
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as exc:
                    st.error(f"Request failed: {exc}")

# Tab 2: Visualizations & Interpretations of >= 3 Features
with tab_viz:
    st.subheader("Dataset Visualizations & Feature Interpretations")

    for img_candidate in [
        "notebook/feature_visualizations.png",
        "feature_visualizations.png",
    ]:
        if os.path.exists(img_candidate):
            st.image(img_candidate, caption="Feature Visualizations", use_container_width=True)
            break
    else:
        st.info("Place `feature_visualizations.png` in `notebook/`")

    for eval_candidate in [
        "notebook/evaluation_metrics.png",
        "notebook/evaluation_results.png",
        "evaluation_metrics.png",
    ]:
        if os.path.exists(eval_candidate):
            st.image(eval_candidate, caption="Model Evaluation Metrics", use_container_width=True)
            break

    st.markdown("### Key Interpretations")
    st.markdown(
        """
        1. **Webpage Aspect Ratio (`width / height`):** Phishing screenshots skew toward higher aspect ratios than legitimate (`other`) screenshots. Many phishing pages capture compact, single-viewport credential-harvesting screens rather than tall, multi-section or scrolled pages.
        2. **Page Brightness (`mean normalized pixel intensity`):** Phishing screenshots tend to have higher average brightness, reflecting stark, high-luminance white/light backgrounds designed to keep attention focused on a login box or impersonated brand mark.
        3. **UI Edge Density vs. Blue Channel Dominance:** Phishing screenshots generally show lower edge density (simpler, sparser UI layouts such as single-form login pages) than denser legitimate sites, while blue channel dominance (`B - R` mean) captures blue-accented visual themes common across impersonated brands and login templates.
        """
    )

# Tab 3: Bulk Data Upload & Retraining Trigger
with tab_retrain:
    st.subheader("Upload Bulk Data & Trigger Model Retraining")
    retrain_class = st.selectbox(
        "Class label for uploaded images",
        options=["phishing", "legitimate"],
        index=0,
    )
    bulk_files = st.file_uploader(
        "Select multiple screenshot images (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bulk_retrain_uploader",
    )

    if st.button("1. Save Uploaded Data for Retraining"):
        if not bulk_files:
            st.warning("Please choose at least one image first.")
        else:
            payload = [
                ("files", (f.name, f.getvalue(), f.type or "image/png"))
                for f in bulk_files
            ]
            try:
                r = requests.post(
                    f"{base_url}/upload-retrain-data",
                    data={"class_label": retrain_class},
                    files=payload,
                    timeout=60,
                )
                if r.ok:
                    st.success(f"Uploaded data saved: {r.json()}")
                else:
                    st.error(f"Upload failed ({r.status_code}): {r.text}")
            except Exception as exc:
                st.error(f"Upload request error: {exc}")

    st.divider()
    col_e, col_lr = st.columns(2)
    with col_e:
        retrain_epochs = st.number_input("Retraining Epochs", min_value=1, max_value=20, value=3)
    with col_lr:
        retrain_lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.6f")

    if st.button("2. Trigger Retraining", type="primary"):
        with st.spinner("Preprocessing uploaded data, retraining model, and reloading weights..."):
            try:
                r = requests.post(
                    f"{base_url}/retrain",
                    params={"epochs": int(retrain_epochs), "learning_rate": float(retrain_lr)},
                    timeout=300,
                )
                if r.ok:
                    st.success("Retraining completed and model reloaded!")
                    st.json(r.json())
                else:
                    st.error(f"Retraining failed ({r.status_code}): {r.text}")
            except Exception as exc:
                st.error(f"Retraining request error: {exc}")

    st.divider()
    st.subheader("Retraining Database")
    if st.button("Refresh Database Status"):
        pass
    try:
        database_response = requests.get(
            f"{base_url}/database/status",
            timeout=5,
        )
        if database_response.ok:
            database_status = database_response.json()
            col_batches, col_files = st.columns(2)
            col_batches.metric("Upload Batches", database_status["batch_count"])
            col_files.metric("Saved Upload Files", database_status["file_count"])
            st.json(database_status["upload_counts_by_class"])
            if database_status["recent_runs"]:
                st.dataframe(database_status["recent_runs"], use_container_width=True)
            else:
                st.caption("No retraining runs recorded yet.")
        else:
            st.warning(
                f"Could not read database status ({database_response.status_code})"
            )
    except Exception as exc:
        st.error(f"Database status request failed: {exc}")