import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import io
import time
import altair as alt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Industrial Machine Fault Prediction", layout="wide")

st.title("⚙️ Industrial Machine Fault Detection System")
st.write("Upload CSV logs for vibration, temperature, pressure, humidity & machine status 🚀")

uploaded_files = st.file_uploader("Upload All CSV Files", type=["csv"], accept_multiple_files=True)

if uploaded_files:

    st.success(f"{len(uploaded_files)} files uploaded!")
    st.write([f.name for f in uploaded_files])

    os.makedirs("data", exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join("data", f.name), "wb") as saved:
            saved.write(f.read())

    csv_files = glob.glob("data/*.csv")

    annotation_file = None
    sensor_files = []
    for file in csv_files:
        df = pd.read_csv(file)
        cols = df.columns.str.lower().tolist()
        if "machine_status" in cols:
            annotation_file = file
        elif "sensor_id" in df.columns:
            sensor_files.append(file)

    if not annotation_file:
        st.error("No annotation file found containing 'machine_status'.")
        st.stop()

    st.info(f"Annotation File Identified: {annotation_file}")
    st.write("Sensor Files Detected:", sensor_files)

    st.subheader("🔄 Merging Dataset...")

    sensor_map = {"vibration": "VIBRATION", "humidity": "HUMIDITY",
                  "pressure": "PRESSURE", "temperature": "TEMPERATURE"}
    dfs = []

    for f in sensor_files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sid = str(df["sensor_id"].iloc[0]).upper()

        sensor_name = None
        for key, match in sensor_map.items():
            if match in sid:
                sensor_name = key
                break

        if sensor_name is None:
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
    st.success("Merged Successfully! 🎯")
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

    # ---------------- Compact Confusion Matrix --------------------
    st.subheader("📉 Confusion Matrix (Compact)")

    y_pred = rf.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 8, "weight": "bold"},
        square=True, linewidths=0.5, linecolor="white"
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    plt.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    st.image(buf.getvalue(), width=250)

    # ---------------- Machine Level Forecast --------------------
    st.subheader("🚨 Machine-Level 10-Day Failure Forecast")

    NUM_MACHINES = 200
    DAYS = 10
    np.random.seed(42)

    base_rows = merged[FEATURES].tail(12).values
    failure_idx = np.where(le.classes_ == "failure")[0]
    failure_idx = failure_idx[0] if len(failure_idx) else None

    results = []
    machine_ids = [f"Machine_{i+1}" for i in range(NUM_MACHINES)]

    # NEW FIXED FAILURE-LOCKING LOGIC
    for m_id in machine_ids:
        has_failed = False
        for d in range(1, DAYS + 1):

            if has_failed:
                results.append({
                    "Machine_ID": m_id,
                    "Day": d,
                    "Predicted_Status": "failure",
                    "Failure_Probability": 1.0
                })
                continue

            base = base_rows[np.random.randint(0, len(base_rows))]
            noisy = base + np.random.normal(0, [0.02, 0.005, 0.002, 0.005])
            scaled = scaler.transform(noisy.reshape(1, -1))

            pred_num = rf.predict(scaled)[0]
            pred_label = le.inverse_transform([pred_num])[0]
            prob_failure = rf.predict_proba(scaled)[0][failure_idx] if failure_idx is not None else 0.0

            if pred_label == "failure":
                has_failed = True

            results.append({
                "Machine_ID": m_id,
                "Day": d,
                "Predicted_Status": "failure" if has_failed else pred_label,
                "Failure_Probability": 1.0 if has_failed else round(float(prob_failure), 4)
            })

    forecast_df = pd.DataFrame(results)

    selected_day = st.selectbox("📅 Select Forecast Day", range(1, DAYS + 1))
    day_df = forecast_df[forecast_df["Day"] == selected_day]

    st.write(f"🔍 Showing Predictions for **Day {selected_day}**")
    st.dataframe(day_df)

    failed = day_df[day_df["Predicted_Status"] == "failure"]
    warning = day_df[day_df["Predicted_Status"] == "warning"]
    normal = day_df[day_df["Predicted_Status"] == "normal"]

    st.subheader("🛑 Machines Predicted to FAIL")
    if len(failed) > 0:
        st.dataframe(failed[["Machine_ID", "Failure_Probability"]])
    else:
        st.info("No Failures detected.")

    st.subheader("⚠️ Warning Machines")
    if len(warning) > 0:
        st.dataframe(warning[["Machine_ID", "Failure_Probability"]])
    else:
        st.info("No machines in WARNING state.")

    st.subheader("✅ Normal Machines")
    if len(normal) > 0:
        st.dataframe(normal[["Machine_ID"]])
    else:
        st.info("No machines operating normally.")

    # ---------------- 10 Day Summary Table --------------------
    st.subheader("📊 10-Day Failure Risk Summary")

    summary_rows = []
    for d in range(1, DAYS + 1):
        ddf = forecast_df[forecast_df["Day"] == d]
        fail_count = int(len(ddf[ddf["Predicted_Status"] == "failure"]))
        mean_prob = round(float(ddf["Failure_Probability"].mean()), 4)
        summary_rows.append({
            "Day": d,
            "Machines_Predicted_Failure": fail_count,
            "Mean_Failure_Probability": mean_prob
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df)

    # BAR CHART – RED WHEN FAILURE OCCURS
    chart = alt.Chart(summary_df).mark_bar().encode(
        x=alt.X("Day:O", title="Day"),
        y=alt.Y("Machines_Predicted_Failure:Q", title="Machines Predicted Failure"),
        color=alt.condition(
            alt.datum.Machines_Predicted_Failure > 0,
            alt.value("red"),
            alt.value("#bdbdbd")
        ),
        tooltip=[
            "Day:O",
            "Machines_Predicted_Failure:Q",
            alt.Tooltip("Mean_Failure_Probability:Q", format=".4f")
        ]
    ).properties(width=700, height=320, title="10-Day Failure Risk Overview")

    st.altair_chart(chart, use_container_width=True)

    # ---------------- Downloads --------------------
    st.download_button(
        "📥 Download Full 10-Day Machine Forecast",
        data=forecast_df.to_csv(index=False),
        file_name="machine_forecast_10days.csv"
    )

    st.download_button(
        "📥 Download Merged Data",
        data=open("merged_data.csv", "rb"),
        file_name="merged_data.csv"
    )
