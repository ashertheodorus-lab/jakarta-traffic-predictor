# Jakarta Traffic Predictor

A simple Streamlit machine-learning app that estimates historical traffic congestion for a Jakarta road location and time of day.

## Project files

```text
jakarta_traffic_predictor/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── jakarta_traffic_long.csv
```

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

## Publish with GitHub and Streamlit Community Cloud

1. Create a new **public** GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the complete `data` folder.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy.

Keep the CSV at exactly `data/jakarta_traffic_long.csv`, because the app loads it from that path.

## Dataset limitation

The supplied dataset has coordinates, road length, historical jam-factor statistics, and a time-period label. It does not contain counts of cars, motorcycles, buses, or trucks. `jam_factor_count` is the number of source observations contributing to an aggregate, not a vehicle count. This version therefore predicts historical jam factor using location, time period, and nearest road length. It is not a live traffic forecast.
