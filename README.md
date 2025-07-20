# Bus_ETA_App
This project predicts bus travel times for an urban transit route using historical trajectory data, traffic conditions (via Google Maps), and real-time weather features (via OpenWeatherMap). A LightGBM-based ensemble model segments traffic levels into low, med, and high, training separate regressors per segment for accurate ETA estimation. 
# 🚌 Bus ETA Prediction Dashboard

This project predicts Estimated Time of Arrival (ETA) for buses on a specific urban route using machine learning and contextual data such as traffic and weather. It includes a full pipeline from data processing to model training and visualization via a Streamlit dashboard.

## 🔍 Project Features

- LightGBM models trained per traffic segment (`low`, `med`, `high`)
- Time-aware features like rolling delays, hour/day cycles, and speed
- Integration with:
  - **Google Maps API** for traffic duration between stops
  - **OpenWeatherMap API** for weather data at bus stops
- Residual and performance analysis by traffic level
- Full interactive dashboard for route analysis

---

## 🚀 Demo

**Streamlit App** (public):
> [https://your-username-your-repo.streamlit.app](#)  
> *(Replace with actual link after deployment)*

---
