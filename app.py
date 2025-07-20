import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import Booster
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import seaborn as snsmlit
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Bus ETA Dashboard", layout="wide")

feature_cols = [
    "prev_arrival_secs","distance_prev_km","speed_kmh",
    "traffic_time_sec","temp_c","rain_flag","humidity",
    "rain_x_traffic","hour_sin","hour_cos","dow_sin","dow_cos",
    "rolling_tt3","delay_trend","travel_time_rainy"
]
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv("outputs/df_with_features_and_segments.csv")
   
    models = {}
    for seg in ["low","med","high"]:
        models[seg] = Booster(model_file=f"models/lgb_{seg}.txt")
    return df, models

df, submodels = load_data_and_models()

seg = st.sidebar.selectbox("Traffic Segment", ["low","med","high"])
trip = st.sidebar.selectbox("Trip ID", df.trip_id.unique())
stop_seq = st.sidebar.slider("Stop sequence", 
                             int(df.stop_sequence.min()), 
                             int(df.stop_sequence.max()))

st.title("🚍 Bus ETA Dashboard")

# 1) Segment MAE
mask = df.traffic_seg_q == seg
y_true = df.loc[mask, "travel_time_rainy"].values
X_seg  = df.loc[mask]
X_feat = X_seg[feature_cols].values
y_pred = submodels[seg].predict(X_feat)
mae_seg = mean_absolute_error(y_true, y_pred)
st.metric(f"MAE for '{seg}' segment", f"{mae_seg:.2f} s")

# 2) Live ETA for the single sample
st.subheader("Live ETA Prediction")
sample = df[(df.trip_id == trip) & (df.stop_sequence == stop_seq)].iloc[0]
pred   = submodels[sample.traffic_seg_q].predict(
    sample[feature_cols].values.reshape(1,-1)
)[0]
col1, col2 = st.columns(2)
col1.metric("🕒 Predicted travel time", f"{pred:.1f} s")
col2.metric("📍 Actual travel time", f"{sample.travel_time_rainy:.1f} s")

st.markdown("---")

# 3) Predicted vs Actual line chart for the full trip
st.subheader(f"Trip {trip}: Predicted vs Actual Travel Time")
trip_df = df[df.trip_id == trip].sort_values("stop_sequence")
trip_df["predicted"] = trip_df.apply(
    lambda r: submodels[r.traffic_seg_q]
                   .predict(r[feature_cols].values.reshape(1,-1))[0],
    axis=1
)
fig = px.line(
    trip_df,
    x="stop_sequence",
    y=["travel_time_rainy","predicted"],
    labels={"value":"Travel Time (s)","stop_sequence":"Stop Sequence"},
    title="Actual vs Predicted Travel Time per Stop"
)
st.plotly_chart(fig, use_container_width=True)

# 4) Data table of this trip
with st.expander("Show trip data table"):
    st.dataframe(trip_df[[
        "stop_sequence","stop_id","travel_time_rainy","predicted",
        "traffic_time_sec","rain_flag"
    ]].reset_index(drop=True), height=300)
    

# 5) (Optional) Residual histogram toggle
if st.sidebar.checkbox("Show residual histogram"):
    residuals = np.abs(trip_df.travel_time_rainy - trip_df.predicted)
    fig2 = px.histogram(
        x=residuals,
        nbins=20,
        log_y=True,
        labels={"x":"Absolute Error (s)"},
        title="Residual Histogram (log scale y-axis)"
    )
    st.plotly_chart(fig2, use_container_width=True)


# 3) Live ETA demo
st.subheader("Live ETA Prediction")
sample = df[(df.trip_id==trip)&(df.stop_sequence==stop_seq)].iloc[0]
pred = submodels[sample.traffic_seg_q].predict(sample[feature_cols].values.reshape(1,-1))[0]
col1, col2 = st.columns(2)
col1.metric("Predicted travel time", f"{pred:.1f} s")
col2.metric("Actual travel time", f"{sample.travel_time_rainy:.1f} s")
