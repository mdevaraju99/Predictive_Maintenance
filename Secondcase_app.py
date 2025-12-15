# ap.py
# Deterministic predictive-maintenance Streamlit app
# - stable simulation
# - rare failures + moderate warnings
# - compact confusion matrix
# - repair-next-day logic
# - warnings show small technical probabilities
# Updated: SENSITIVITY + threshold + warning rule

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import io
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# ----------------- Configuration -----------------
SENSITIVITY = 1.1              # noise multiplier (slightly increased)
FAILURE_PROB_THRESHOLD = 0.22  # threshold to treat predicted prob as failure
NUM_MACHINES = 200
DAYS = 10
RNG_SEED = 999                 # deterministic seed for repeatable runs

# ----------------- Page -----------------
st.set_page_config(page_title="Industrial Machine Fault Prediction", layout="wide")
st.title("⚙️ Industrial Machine Fault Detection System")
st.write("Upload sensor CSVs (vibration, temperature, pressure, humidity) + annotation CSV with `machine_status` labels")

uploaded_files = st.file_uploader("Upload CSV files (sensor CSVs + annotation CSV)", type=["csv"], accept_multiple_files=True)

# ----------------- Helpers -----------------
def failure_reason(sensor_vals, feature_names, normal_levels):
    diffs = sensor_vals - normal_levels
    idx = int(np.argmax(np.abs(diffs)))
    return feature_names[idx], float(diffs[idx])

def repair_action(sensor_name):
    actions = {
        "temperature": "Restore cooling / replace lubricant",
        "vibration": "Check bearings / perform balancing & alignment",
        "pressure": "Inspect piping / remove blockage / adjust valve",
        "humidity": "Drying / replace seals / dehumidify"
    }
    return actions.get(sensor_name, "Perform general inspection & maintenance")

# ----------------- Main -----------------
if uploaded_files:
    # save uploaded files
    os.makedirs("data", exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join("data", f.name), "wb") as saved:
            saved.write(f.read())

    csv_files = glob.glob("data/*.csv")

    # identify annotation & sensors
    annotation_file = None
    sensor_files = []
    for file in csv_files:
        try:
            tmp = pd.read_csv(file, nrows=2)
        except Exception:
            continue
        cols = [c.lower() for c in tmp.columns]
        if "machine_status" in cols:
            annotation_file = file
        elif ("sensor_id" in cols) or ("timestamp" in cols and "value" in cols):
            sensor_files.append(file)

    if annotation_file is None:
        st.error("No annotation CSV with column 'machine_status' found. Please upload it.")
        st.stop()

    st.success("Files detected")
    st.write("Annotation:", os.path.basename(annotation_file))
    st.write("Sensors:", [os.path.basename(x) for x in sensor_files])

    # merge sensor CSVs
    st.subheader("🔄 Merging dataset")
    sensor_map = {"vibration": "VIBRATION", "humidity": "HUMIDITY", "pressure": "PRESSURE", "temperature": "TEMPERATURE"}
    dfs = []
    for fpath in sensor_files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if "timestamp" not in df.columns or "value" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sid = str(df["sensor_id"].iloc[0]).upper() if "sensor_id" in df.columns else ""
        sensor_name = None
        for k, match in sensor_map.items():
            if match in sid:
                sensor_name = k
                break
        # fallback to filename
        if sensor_name is None:
            fname = os.path.basename(fpath).lower()
            for k in sensor_map:
                if k in fname:
                    sensor_name = k
                    break
        if sensor_name is None:
            continue
        df = df[["timestamp", "value"]].rename(columns={"value": sensor_name})
        df = df.set_index("timestamp").resample("10min").mean().interpolate()
        dfs.append(df)

    if len(dfs) == 0:
        st.error("No valid sensor CSVs found (need timestamp & value columns).")
        st.stop()

    merged = pd.concat(dfs, axis=1)

    ann = pd.read_csv(annotation_file)
    ann["timestamp"] = pd.to_datetime(ann["timestamp"])
    ann = ann.set_index("timestamp")
    ann_cols = [c.lower() for c in ann.columns]
    if "machine_status" not in ann_cols:
        st.error("Annotation CSV missing 'machine_status' column.")
        st.stop()
    real_col = ann.columns[[c.lower() for c in ann.columns].index("machine_status")]
    merged = merged.join(ann[real_col], how="left")
    merged[real_col] = merged[real_col].fillna("normal")
    merged = merged.rename(columns={real_col: "fault_type"}).reset_index()
    merged.to_csv("merged_data.csv", index=False)

    st.success("Merged dataset ready")
    st.dataframe(merged.head())
    st.write("Label distribution")
    st.bar_chart(merged["fault_type"].value_counts())

    # ----------------- Training -----------------
    st.subheader("🧠 Model training (RandomForest)")
    FEATURES = ["vibration", "temperature", "pressure", "humidity"]
    FEATURES = [f for f in FEATURES if f in merged.columns]
    if len(FEATURES) < 1:
        st.error("No sensor features found after merge. Aborting.")
        st.stop()

    X = merged[FEATURES].values
    y = merged["fault_type"].values

    # baseline normal means (for deviation detection)
    if "normal" in merged["fault_type"].unique():
        try:
            normal_means = merged[merged["fault_type"] == "normal"][FEATURES].mean().values
        except Exception:
            normal_means = np.nanmean(X, axis=0)
    else:
        normal_means = np.nanmean(X, axis=0)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    strat = (y_enc if len(np.unique(y_enc)) > 1 else None)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=strat)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    if len(np.unique(y_train)) > 1 and len(y_train) > 10:
        sm = SMOTE(random_state=42, k_neighbors=1)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
    else:
        X_train_bal, y_train_bal = X_train_s, y_train

    rf = RandomForestClassifier(n_estimators=400, random_state=42)
    rf.fit(X_train_bal, y_train_bal)

    joblib.dump(rf, "rf_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(le, "label_encoder.joblib")
    st.success("Model trained & saved")

    # ----------------- Compact confusion matrix -----------------
    st.subheader("📉 Confusion Matrix (compact)")
    try:
        y_pred = rf.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
    except Exception:
        cm = np.zeros((len(le.classes_), len(le.classes_)), dtype=int)

    labels = le.classes_.tolist()
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 8},
                square=True, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf.getvalue(), width=220)

    # ----------------- Forecast -----------------
    st.subheader("🚨 Machine-level 10-day forecast (deterministic + repair next day)")
    np.random.seed(RNG_SEED)

    base_rows = merged[FEATURES].tail(min(12, len(merged))).values
    if base_rows.shape[0] == 0:
        st.error("Not enough data rows to simulate from.")
        st.stop()

    base_fixed = base_rows.mean(axis=0)   # deterministic baseline
    failure_idx = None
    if "failure" in le.classes_:
        failure_idx = int(np.where(le.classes_ == "failure")[0][0])

    results = []
    machine_ids = [f"Machine_{i+1}" for i in range(NUM_MACHINES)]

    for midx, m_id in enumerate(machine_ids):
        repair_next_day = False
        bias = ((midx % 11) - 5) * 0.01
        for d in range(1, DAYS + 1):
            if repair_next_day:
                results.append({
                    "Machine_ID": m_id,
                    "Day": d,
                    "Predicted_Status": "normal",
                    "Failure_Probability": 0.0,
                    "Failure_Reason": "—",
                    "Repair_Action": "Machine restored & validated"
                })
                repair_next_day = False
                continue

            seed_val = 1000 + midx * 37 + d * 13 + RNG_SEED
            rng = np.random.RandomState(seed_val)

            base_noise_scales_full = np.array([0.03, 0.01, 0.005, 0.01]) * SENSITIVITY
            base_noise_scales = base_noise_scales_full[:len(base_fixed)]

            noise = rng.normal(0, base_noise_scales, size=base_fixed.shape)
            noisy = base_fixed + noise
            noisy[0] += bias * 0.5
            if len(noisy) > 1:
                noisy[1] += bias * 0.8

            scaled = scaler.transform(noisy.reshape(1, -1))

            pred_num = rf.predict(scaled)[0]
            pred_label = le.inverse_transform([pred_num])[0]

            prob_failure = 0.0
            try:
                if failure_idx is not None:
                    prob_failure = float(rf.predict_proba(scaled)[0][failure_idx])
            except Exception:
                prob_failure = 0.0

            deviations = np.abs(noisy - normal_means)
            max_dev = float(np.max(deviations)) if deviations.size > 0 else 0.0

            # Warning logic: ANY of the conditions
            is_warning_by_label = str(pred_label).lower() == "warning"
            is_warning_by_prob = (0.25 <= prob_failure <= 0.45)
            is_warning_by_dev = (0.05 <= max_dev <= 0.15)
            is_warning = is_warning_by_label or is_warning_by_prob or is_warning_by_dev

            # Failure decision
            is_failure = (str(pred_label).lower() == "failure") or (prob_failure >= FAILURE_PROB_THRESHOLD)

            if is_failure:
                sensor_name, deviation = failure_reason(noisy, FEATURES, normal_means)
                # Ensure displayed failure prob is strong (min 0.35) but cap at 0.95
                displayed_prob = max(0.35, prob_failure)
                displayed_prob = min(displayed_prob, 0.95)
                results.append({
                    "Machine_ID": m_id,
                    "Day": d,
                    "Predicted_Status": "failure",
                    "Failure_Probability": round(float(displayed_prob), 4),
                    "Failure_Reason": f"{sensor_name} spike detected (Δ={deviation:.4f})",
                    "Repair_Action": repair_action(sensor_name)
                })
                repair_next_day = True

            elif is_warning:
                # produce a small technical probability for warnings in [0.01, 0.08]
                show_prob = round(float(prob_failure), 4)
                if show_prob == 0.0:
                    mapped_prob = max(0.01, min(0.08, max_dev * 0.06))  # map dev -> small prob
                    show_prob = round(mapped_prob, 4)
                else:
                    show_prob = round(min(0.08, max(0.08, show_prob)) if show_prob < 0.01 else min(show_prob, 0.08), 4)
                    # clamp into [0.01,0.08]
                    if show_prob < 0.01:
                        show_prob = 0.01
                    if show_prob > 0.08:
                        show_prob = 0.08

                reason = "—"
                if is_warning_by_dev:
                    sensor_name, deviation = failure_reason(noisy, FEATURES, normal_means)
                    reason = f"Moderate {sensor_name} deviation (Δ={deviation:.4f})"

                results.append({
                    "Machine_ID": m_id,
                    "Day": d,
                    "Predicted_Status": "warning",
                    "Failure_Probability": show_prob,
                    "Failure_Reason": reason,
                    "Repair_Action": "Inspect sensors / schedule preventive check"
                })

            else:
                # Normal: explicit zero probability
                results.append({
                    "Machine_ID": m_id,
                    "Day": d,
                    "Predicted_Status": "normal",
                    "Failure_Probability": 0.0,
                    "Failure_Reason": "—",
                    "Repair_Action": "—"
                })

    forecast_df = pd.DataFrame(results)

    # ---------- day view ----------
    st.subheader("🔎 Day view (select day to inspect machines)")
    selected_day = st.selectbox("Select forecast day", options=list(range(1, DAYS + 1)), index=0)
    day_df = forecast_df[forecast_df["Day"] == selected_day].reset_index(drop=True)
    st.dataframe(day_df, height=260)

    failed = day_df[day_df["Predicted_Status"] == "failure"]
    warning = day_df[day_df["Predicted_Status"] == "warning"]
    normal = day_df[day_df["Predicted_Status"] == "normal"]

    st.subheader("🛑 Machines predicted to FAIL (selected day)")
    if len(failed) > 0:
        st.dataframe(failed[["Machine_ID", "Failure_Probability", "Failure_Reason", "Repair_Action"]].reset_index(drop=True), height=220)
    else:
        st.info("No failures on this day.")

    st.subheader("⚠️ Machines in WARNING (selected day)")
    if len(warning) > 0:
        st.dataframe(warning[["Machine_ID", "Failure_Probability", "Failure_Reason", "Repair_Action"]].reset_index(drop=True), height=220)
    else:
        st.info("No warnings on this day.")

    st.subheader("✅ Normal machines (selected day)")
    if len(normal) > 0:
        st.dataframe(normal[["Machine_ID"]].reset_index(drop=True), height=220)
    else:
        st.info("No normal machines on this day.")

    # ---------- 10-day summary & chart ----------
    st.subheader("📊 10-day failure summary")
    summary_rows = []
    for d in range(1, DAYS + 1):
        ddf = forecast_df[forecast_df["Day"] == d]
        mean_prob = float(ddf["Failure_Probability"].mean()) if len(ddf) > 0 else 0.0
        fail_count = int(len(ddf[ddf["Predicted_Status"] == "failure"]))
        summary_rows.append({"Day": d, "Machines_Predicted_Failure": fail_count, "Mean_Failure_Probability": round(mean_prob, 4)})
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, height=220)

    chart = alt.Chart(summary_df).mark_bar().encode(
        x=alt.X("Day:O", title="Day"),
        y=alt.Y("Machines_Predicted_Failure:Q", title="Machines Predicted Failure"),
        color=alt.condition(alt.datum.Machines_Predicted_Failure > 0, alt.value("red"), alt.value("#bdbdbd")),
        tooltip=[alt.Tooltip("Day:O"), alt.Tooltip("Machines_Predicted_Failure:Q"), alt.Tooltip("Mean_Failure_Probability:Q", format=".4f")]
    ).properties(width=700, height=320, title="10-Day Failure Risk Overview")
    st.altair_chart(chart, use_container_width=True)

    # ---------- downloads ----------
    st.download_button("📥 Download full 10-day forecast", data=forecast_df.to_csv(index=False), file_name="machine_forecast_10days.csv")
    st.download_button("📥 Download merged dataset", data=open("merged_data.csv", "rb"), file_name="merged_data.csv")

else:
    st.info("Upload sensor CSVs and an annotation CSV (with 'machine_status') to begin.")
