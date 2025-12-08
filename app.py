import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Industrial Machine Fault Prediction", layout="wide")

st.title("⚙️ Industrial Machine Fault Detection System")
st.write("Upload CSV logs for vibration, temperature, pressure, humidity & machine status 🚀")

# ---------------- Upload Section --------------------
uploaded_files = st.file_uploader("Upload All CSV Files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded!")
    st.write([f.name for f in uploaded_files])

    # Save to temp directory
    os.makedirs("data", exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join("data", f.name), "wb") as saved:
            saved.write(f.read())

    csv_files = glob.glob("data/*.csv")

    # ---------------- Find annotation CSV --------------------
    annotation_file = None
    sensor_files = []
    for file in csv_files:
        df = pd.read_csv(file)
        cols = df.columns.str.lower().tolist()
        if "machine_status" in cols:
            annotation_file = file
        else:
            if "sensor_id" in df.columns:
                sensor_files.append(file)

    if not annotation_file:
        st.error("No machine_status annotation file found.")
        st.stop()

    st.info(f"Annotation File Identified: **{annotation_file}**")
    st.write("Sensor Files:", sensor_files)

    # ---------------- Merge Logic --------------------
    st.subheader("🔄 Merging Dataset...")

    sensor_map = {"vibration": "VIBRATION", "humidity": "HUMIDITY", "pressure": "PRESSURE", "temperature": "TEMPERATURE"}
    dfs = []

    for f in sensor_files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sid = df["sensor_id"].iloc[0].upper()

        sensor_name = None
        for key, match in sensor_map.items():
            if match in sid:
                sensor_name = key
                break
        if not sensor_name:
            continue

        df = df[["timestamp", "value"]].rename(columns={"value": sensor_name})
        df = df.set_index("timestamp").resample("10min").mean().interpolate()
        dfs.append(df)

    merged = pd.concat(dfs, axis=1)

    ann = pd.read_csv(annotation_file)
    ann["timestamp"] = pd.to_datetime(ann["timestamp"])
    ann = ann.set_index("timestamp")

    merged = merged.join(ann["machine_status"], how="left")
    merged["machine_status"] = merged["machine_status"].fillna("normal")
    merged = merged.rename(columns={"machine_status": "fault_type"})
    merged = merged.reset_index()

    merged.to_csv("merged_data.csv", index=False)
    st.success("Merged Successfully!")
    st.dataframe(merged.head())

    st.write("Label Distribution:")
    st.bar_chart(merged["fault_type"].value_counts())

    # ---------------- Train Model --------------------
    st.subheader("🧠 Model Training")

    FEATURES = ["vibration", "temperature", "pressure", "humidity"]
    X = merged[FEATURES].values
    y = merged["fault_type"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)

    rf = RandomForestClassifier(n_estimators=400, random_state=42)
    rf.fit(X_train_bal, y_train_bal)

    joblib.dump(rf, "rf_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(le, "label_encoder.joblib")

    st.success("Model Trained & Saved Successfully!")

    # ---------------- Evaluation (Confusion Matrix Only) --------------------
    y_pred = rf.predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="viridis", values_format="d")

    ax.set_title("Confusion Matrix - Random Forest Fault Classification", fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    st.pyplot(fig)

    # ---------------- Prediction Simulation --------------------
    st.subheader("🚨 10-Day Failure Forecast Simulation")

    NUM_MACHINES = 200
    DAYS = 10
    np.random.seed(42)

    base_rows = merged[FEATURES].tail(12).values

    failure_idx = np.where(le.classes_ == "failure")[0]
    failure_idx = failure_idx[0] if len(failure_idx) else None

    rows = []
    for d in range(1, DAYS+1):
        probs_list = []
        for _ in range(NUM_MACHINES):
            base = base_rows[np.random.randint(0, len(base_rows))]
            noisy = base + np.random.normal(0, [0.02,0.005,0.002,0.005])
            scaled = scaler.transform(noisy.reshape(1,-1))
            prob = rf.predict_proba(scaled)[0][failure_idx] if failure_idx is not None else 0
            probs_list.append(prob)

        rows.append({
            "Day": d,
            "Mean Failure Probability": round(float(np.mean(probs_list)), 4),
            "Machines Predicted Failure": int(np.sum(np.array(probs_list) >= 0.2))
        })

    forecast_df = pd.DataFrame(rows)
    st.dataframe(forecast_df)

    st.line_chart(forecast_df.set_index("Day")[["Machines Predicted Failure"]])

    st.download_button("📥 Download Merged Data", data=open("merged_data.csv","rb"), file_name="merged_data.csv")
