# ============================================================
# S.A.F.E. — Smart AI For Everyone (Ghaziabad Safe Route v2.1)
# ============================================================
# ✅ Fully functional AI risk integration
# ✅ Fixed feature_order scope issue
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import requests, polyline, math, os, joblib
from sklearn.neighbors import KDTree
from streamlit.components.v1 import html
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="S.A.F.E. — AI Safe Route Planner", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------
# Utilities
# ---------------------------------------------
def haversine_m(p1, p2):
    R = 6371000.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def osrm_route(origin, destination, alternatives=False):
    o = f"{origin[1]},{origin[0]}"
    d = f"{destination[1]},{destination[0]}"
    url = f"http://router.project-osrm.org/route/v1/driving/{o};{d}"
    params = {"overview":"full","alternatives":"true" if alternatives else "false","geometries":"polyline","steps":"false"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    rr = r.json()
    routes = []
    for route in rr.get("routes", []):
        coords = polyline.decode(route["geometry"])
        routes.append({
            "coords": coords,
            "distance": route["distance"],
            "duration": route["duration"]
        })
    return routes

def folium_map(center, heat_points, best_coords, alt_coords=[]):
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter")
    if heat_points:
        HeatMap(heat_points, radius=12, blur=18, max_zoom=1, min_opacity=0.3).add_to(m)
    for alt in alt_coords:
        folium.PolyLine(locations=alt, weight=4, color="#888888", opacity=0.35).add_to(m)
    folium.PolyLine(best_coords, weight=8, color="#00ffd5", opacity=0.95).add_to(m)
    folium.CircleMarker(best_coords[0], radius=6, color="#ffffff", fill=True, fill_color="#00ff99").add_to(m)
    folium.CircleMarker(best_coords[-1], radius=6, color="#ffffff", fill=True, fill_color="#ff4666").add_to(m)
    return m

# ---------------------------------------------
# Header
# ---------------------------------------------
st.markdown("<h1 style='color:#00fff7'>S.A.F.E. — Smart AI For Everyone</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#9fbfff'>AI-Powered Safe Route Planner (Ghaziabad)</h4>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1,2])

# ---------------------------------------------
# Load model, scaler, and features
# ---------------------------------------------
MODEL_PATH = "crime_hotspot_model_v7.pkl"
SCALER_PATH = "crime_scaler_v7.pkl"
FEATURES_PATH = "features_v7.txt"
use_model = False
feature_order = []

if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH) as f:
            feature_order = [line.strip() for line in f if line.strip()]
        st.success("✅ Model, Scaler & Feature list loaded successfully — AI risk enabled.")
        use_model = True
    except Exception as e:
        st.warning(f"⚠️ Error loading model/scaler: {e}. Using fallback historical risk.")
else:
    st.warning("⚠️ Model/scaler/features file missing. Using fallback count-based risk.")

# ---------------------------------------------
# Sidebar Inputs
# ---------------------------------------------
with col1:
    st.header("Route Configuration")
    origin_lat = st.number_input("Origin latitude", value=28.6692, format="%.6f")
    origin_lon = st.number_input("Origin longitude", value=77.4538, format="%.6f")
    dest_lat = st.number_input("Destination latitude", value=28.7041, format="%.6f")
    dest_lon = st.number_input("Destination longitude", value=77.4520, format="%.6f")
    time_of_day = st.time_input("Time (for context)", value=pd.to_datetime("20:00").time())
    gender = st.selectbox("Gender", ["any", "female", "male"], index=1)
    alpha = st.slider("Safety vs Time tradeoff", 0.0, 1.0, 0.8, 0.05)
    show_heat = st.checkbox("Show heatmap overlay", True)
    show_alts = st.checkbox("Show alternative routes", True)
    run_btn = st.button("🔍 Find Safest Route")

# ---------------------------------------------
# Build AI cell risk map
# ---------------------------------------------
@st.cache_data(ttl=300)
def build_cell_risk(use_ai=True, feature_order=None):
    df = pd.read_csv("ghaziabad_synthetic_crimes.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['month'] = df['date_time'].dt.month
    df['weekday'] = df['date_time'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

    crime_dummies = pd.get_dummies(df['crime_type'], prefix="crime")
    df = pd.concat([df, crime_dummies], axis=1)

    lat_bins = np.linspace(df.latitude.min(), df.latitude.max(), 15)
    lon_bins = np.linspace(df.longitude.min(), df.longitude.max(), 15)
    df['lat_bin'] = pd.cut(df.latitude, lat_bins, labels=False)
    df['lon_bin'] = pd.cut(df.longitude, lon_bins, labels=False)
    df['cell'] = df['lat_bin'].astype(str) + "_" + df['lon_bin'].astype(str)

    agg = df.groupby(['cell', df['date_time'].dt.date]).agg({
        'crime_type':'count',
        'severity':'mean',
        'hour':'mean',
        'month':'first',
        'weekday':'first',
        'is_weekend':'first',
        **{col: 'sum' for col in crime_dummies.columns}
    }).reset_index().rename(columns={'crime_type':'crime_count','hour':'avg_hour'})

    agg['lag1'] = agg.groupby('cell')['crime_count'].shift(1).fillna(0)
    agg['lag2'] = agg.groupby('cell')['crime_count'].shift(2).fillna(0)
    agg['rolling_3'] = agg.groupby('cell')['crime_count'].rolling(3).mean().reset_index(level=0, drop=True).fillna(0)
    agg['month_sin'] = np.sin(2 * np.pi * agg['month'] / 12)
    agg['month_cos'] = np.cos(2 * np.pi * agg['month'] / 12)
    agg['weekday_sin'] = np.sin(2 * np.pi * agg['weekday'] / 7)
    agg['weekday_cos'] = np.cos(2 * np.pi * agg['weekday'] / 7)

    latest_date = agg['date_time'].max()
    agg_latest = agg[agg['date_time'] == latest_date].copy().fillna(0)

    # Fill missing columns
    for cf in feature_order:
        if cf not in agg_latest.columns:
            agg_latest[cf] = 0

    if use_ai:
        X = agg_latest.reindex(columns=feature_order, fill_value=0)
        X_scaled = scaler.transform(X)
        agg_latest['pred_proba'] = model.predict_proba(X_scaled)[:,1]
    else:
        maxc = agg_latest['crime_count'].max() or 1
        agg_latest['pred_proba'] = agg_latest['crime_count'] / maxc

    cent = df.groupby('cell')[['latitude','longitude']].mean().reset_index()
    cell_df = cent.merge(agg_latest[['cell','pred_proba']], on='cell', how='right').fillna(0)
    return cell_df[['cell','latitude','longitude','pred_proba']]

cell_df = build_cell_risk(use_ai=use_model, feature_order=feature_order)
centroids = cell_df[['latitude','longitude']].values
risk_vals = cell_df['pred_proba'].values
kd = KDTree(centroids, metric='euclidean')

def score_route(coords, gender="any"):
    seg_risks, seg_lengths = [], []
    for i in range(len(coords)-1):
        a, b = coords[i], coords[i+1]
        mid = [(a[0]+b[0])/2, (a[1]+b[1])/2]
        _, idx = kd.query([mid], k=1)
        r = float(risk_vals[idx[0][0]])
        if gender == "female":
            r = min(1.0, r * 1.25)
        seg_risks.append(r)
        seg_lengths.append(haversine_m(a,b))
    if not seg_lengths:
        return 0.0
    return sum(r*l for r,l in zip(seg_risks, seg_lengths)) / sum(seg_lengths)

# ---------------------------------------------
# Route generation and scoring
# ---------------------------------------------
with col1:
    if run_btn:
        try:
            origin = (origin_lat, origin_lon)
            dest = (dest_lat, dest_lon)
            routes = osrm_route(origin, dest, alternatives=show_alts)
            if not routes:
                st.error("No route found. Try different points.")
            else:
                scored = []
                for r in routes:
                    risk = score_route(r['coords'], gender)
                    dur_norm = r['duration'] / (r['duration'] + 60)
                    score = alpha * risk + (1-alpha) * dur_norm
                    scored.append({"route": r, "risk": risk, "score": score})
                best = sorted(scored, key=lambda x: x["score"])[0]
                st.success(f"✅ Safest route found — Risk={best['risk']:.3f}, Distance={best['route']['distance']:.0f}m, Duration={best['route']['duration']/60:.1f}min")

                summary = pd.DataFrame([{"distance_m":s['route']['distance'], "duration_min":s['route']['duration']/60.0, "risk":s['risk'], "score":s['score']} for s in scored])
                st.table(summary.style.format({"distance_m":"{:.0f}","duration_min":"{:.1f}","risk":"{:.3f}","score":"{:.3f}"}))

                heat_points = list(zip(cell_df.latitude, cell_df.longitude, cell_df.pred_proba)) if show_heat else None
                alt_coords = [s['route']['coords'] for s in scored[1:]] if show_alts else []
                m = folium_map([(origin_lat+dest_lat)/2,(origin_lon+dest_lon)/2], heat_points, best['route']['coords'], alt_coords)
                html(m._repr_html_(), height=700)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------------------------------------
# Right side: initial map
# ---------------------------------------------
with col2:
    st.header("Citywide Risk Heatmap")
    m0 = folium.Map(location=[28.67,77.45], zoom_start=12, tiles="CartoDB dark_matter")
    if not cell_df.empty:
        HeatMap(list(zip(cell_df.latitude, cell_df.longitude, cell_df.pred_proba)),
                radius=12, blur=18, max_zoom=1, min_opacity=0.3).add_to(m0)
    html(m0._repr_html_(), height=700)

st.markdown("---")
st.markdown("<h4 style='color:#00fff7'>⚙️ Powered by AI Crime Prediction Model (v7)</h4>", unsafe_allow_html=True)
st.caption("S.A.F.E. computes real-time cell risks using trained XGBoost model. Predictions are probabilistic and for awareness only.")
