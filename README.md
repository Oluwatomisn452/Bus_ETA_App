# Bus_ETA_App
This project predicts bus travel times for an urban transit route using historical trajectory data, traffic conditions (via Google Maps), and real-time weather features (via OpenWeatherMap). A LightGBM-based ensemble model segments traffic levels into low, med, and high, training separate regressors per segment for accurate ETA estimation. 
