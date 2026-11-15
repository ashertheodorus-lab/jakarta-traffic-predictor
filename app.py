from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "jakarta_traffic_long.csv"
TARGET = "jam_factor_mean"
PERIODS = {
    "Night (00:00–06:00)": "night",
    "Morning peak (06:00–09:00)": "morning_peak",
    "Morning off-peak (09:00–12:00)": "morning_offpeak",
    "Lunch hours (12:00–14:00)": "lunch_hours",
    "Afternoon off-peak (14:00–16:00)": "afternoon_offpeak",
    "Evening peak (16:00–19:00)": "evening_peak",
    "Evening off-peak (19:00–22:00)": "evening_offpeak",
    "Late night (22:00–00:00)": "late_night",
}
PERIOD_ORDER = list(PERIODS.values())
PERIOD_TO_CODE = {period: i for i, period in enumerate(PERIOD_ORDER)}
FEATURES = ["longitude", "latitude", "road_length_km", "period_code"]

st.set_page_config(page_title="Jakarta Traffic Predictor", page_icon="🚦", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem;}
      div[data-testid="stMetric"] {background: #f5f7fb; border: 1px solid #e5eaf1;
        padding: 14px 16px; border-radius: 12px;}
      div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius: 9px; font-weight: 650; min-height: 2.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Make sure data/jakarta_traffic_long.csv is included in your GitHub repository."
        )
    data = pd.read_csv(DATA_PATH)
    required = {"fid", "longitude", "latitude", "road_length_km", TARGET,
                "jam_factor_std", "jam_factor_count", "period"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    data["period_code"] = data["period"].map(PERIOD_TO_CODE)
    data = data.dropna(subset=FEATURES + [TARGET, "fid"])
    return data


@st.cache_resource
def train_model():
    data = load_data()
    x = data[FEATURES].astype(float)
    y = data[TARGET].astype(float)

    # Group by road ID so one road segment cannot appear in both train and test sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=data["fid"]))

    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, learning_rate=0.08,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    validation_prediction = model.predict(x.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], validation_prediction)

    # Refit the final predictor on all available records.
    final_model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, learning_rate=0.08,
        l2_regularization=1.0, random_state=42,
    )
    final_model.fit(x, y)
    return final_model, float(mae)


def haversine_km(lat, lon, latitudes, longitudes):
    earth_radius = 6371.0
    lat1, lon1 = np.radians(float(lat)), np.radians(float(lon))
    lat2, lon2 = np.radians(np.asarray(latitudes)), np.radians(np.asarray(longitudes))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def traffic_label(score):
    if score < 1:
        return "Free flow"
    if score < 3:
        return "Light traffic"
    if score < 6:
        return "Moderate traffic"
    if score < 8:
        return "Heavy traffic"
    return "Severe congestion"


try:
    data = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

st.title("🚦 Jakarta Traffic Predictor")
st.write("Estimate historical traffic congestion for a Jakarta road location and time of day.")
st.caption("Machine-learning demonstration · Uses the supplied historical road-segment dataset")

# Quick overview
c1, c2, c3 = st.columns(3)
c1.metric("Road segments", f"{data['fid'].nunique():,}")
c2.metric("Dataset records", f"{len(data):,}")
c3.metric("Time periods", f"{data['period'].nunique()}")

with st.expander("Important: what this model can and cannot predict"):
    st.write(
        "The dataset contains road coordinates, road length, time period, and historical jam-factor "
        "statistics. It does **not** contain the number of cars, motorcycles, buses, or trucks. "
        "Therefore, this version predicts a historical congestion score from location, time period, "
        "and the nearest road segment's length. It does not estimate live traffic or vehicle counts."
    )

with st.sidebar:
    st.header("Prediction inputs")
    st.caption("Enter coordinates within the area covered by the dataset.")
    lat_min, lat_max = float(data.latitude.min()), float(data.latitude.max())
    lon_min, lon_max = float(data.longitude.min()), float(data.longitude.max())
    with st.form("prediction_form"):
        latitude = st.number_input(
            "Latitude", min_value=lat_min, max_value=lat_max,
            value=float(data.latitude.median()), step=0.001, format="%.6f"
        )
        longitude = st.number_input(
            "Longitude", min_value=lon_min, max_value=lon_max,
            value=float(data.longitude.median()), step=0.001, format="%.6f"
        )
        selected_period_label = st.selectbox("Time of day", list(PERIODS.keys()), index=5)
        submitted = st.form_submit_button("Predict traffic", type="primary", use_container_width=True)
    st.caption(f"Latitude range: {lat_min:.3f} to {lat_max:.3f}")
    st.caption(f"Longitude range: {lon_min:.3f} to {lon_max:.3f}")

st.subheader("Traffic prediction")
if submitted:
    with st.spinner("Training the model and predicting congestion..."):
        model, validation_mae = train_model()
        selected_period = PERIODS[selected_period_label]

        # Use one row per road segment to find the closest mapped road.
        roads = data.drop_duplicates("fid").copy()
        distances = haversine_km(
            latitude, longitude, roads.latitude.to_numpy(), roads.longitude.to_numpy()
        )
        nearest_idx = int(np.argmin(distances))
        nearest_road = roads.iloc[nearest_idx]
        nearest_distance = float(distances[nearest_idx])

        model_input = pd.DataFrame([{
            "longitude": float(longitude),
            "latitude": float(latitude),
            "road_length_km": float(nearest_road.road_length_km),
            "period_code": float(PERIOD_TO_CODE[selected_period]),
        }], columns=FEATURES)
        prediction = float(np.clip(model.predict(model_input)[0], 0, 10))
        category = traffic_label(prediction)

        a, b, c = st.columns(3)
        a.metric("Predicted jam factor", f"{prediction:.2f} / 10")
        b.metric("Estimated traffic level", category)
        c.metric("Nearest road segment", f"#{int(nearest_road.fid)}")
        st.progress(min(max(prediction / 10, 0), 1), text=f"Congestion score: {prediction:.2f} / 10")
        st.caption(
            f"Period: **{selected_period_label}** · Nearest mapped road: **{nearest_distance:.2f} km** "
            f"away · Road length: **{float(nearest_road.road_length_km):.2f} km**"
        )

        same_period = data[data.period == selected_period].copy()
        same_period["distance_km"] = haversine_km(
            latitude, longitude, same_period.latitude.to_numpy(), same_period.longitude.to_numpy()
        )
        nearby = same_period.nsmallest(5, "distance_km").copy()
        nearby["traffic_level"] = nearby[TARGET].map(traffic_label)

        left, right = st.columns([1.2, 0.8])
        with left:
            st.markdown("#### Nearby road segments")
            display = nearby[["fid", "distance_km", TARGET, "traffic_level"]].rename(columns={
                "fid": "Road ID", "distance_km": "Distance (km)",
                TARGET: "Historical jam factor", "traffic_level": "Traffic level",
            })
            st.dataframe(display.style.format({
                "Distance (km)": "{:.3f}", "Historical jam factor": "{:.2f}"
            }), use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Congestion scale")
            st.write("- **0–1:** Free flow")
            st.write("- **1–3:** Light traffic")
            st.write("- **3–6:** Moderate traffic")
            st.write("- **6–8:** Heavy traffic")
            st.write("- **8–10:** Severe congestion")

        st.metric("Model validation MAE", f"{validation_mae:.3f} jam-factor points")
        st.caption(
            "Validation holds out entire road segments from training. MAE is the average absolute "
            "prediction error on that test split; it is not a guarantee of live-traffic accuracy."
        )
        with st.expander("View nearby roads on a map"):
            map_data = nearby[["latitude", "longitude"]].copy()
            map_data.loc[len(map_data)] = [latitude, longitude]
            st.map(map_data, zoom=11)
else:
    st.info("Choose a location and time period in the sidebar, then click **Predict traffic**.")

st.divider()
st.subheader("Traffic analytics")
period_summary = (
    data.groupby("period", as_index=False)[TARGET].mean()
    .assign(period=lambda frame: pd.Categorical(frame.period, categories=PERIOD_ORDER, ordered=True))
    .sort_values("period")
    .set_index("period")
)
st.write("Average historical jam factor across the eight daily time periods:")
st.bar_chart(period_summary[TARGET])

with st.expander("About the dataset and machine-learning model"):
    st.markdown(
        """
        - **Dataset:** `jakarta_traffic_long.csv`
        - **Records:** 116,392 rows covering 14,549 road segments and eight time periods.
        - **Inputs:** longitude, latitude, nearest road length, and selected time period.
        - **Prediction target:** `jam_factor_mean`, the historical mean jam factor.
        - **Algorithm:** `HistGradientBoostingRegressor`.
        - **Validation:** 20% of road-segment IDs are held out from training.
        - **Limitation:** no vehicle counts or live traffic feed are included in this dataset.
        """
    )
st.caption("Educational prototype. This is a historical-pattern estimate, not a live navigation service.")
