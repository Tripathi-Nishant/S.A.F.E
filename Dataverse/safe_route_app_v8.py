# ================================================================
#  S.A.F.E. — Smart AI For Everyone (v8 FINAL — Permanent Fix)
# ================================================================
# 🚀 Fixes:
# ✅ Auto-matches column names (case + underscores)
# ✅ Ensures every model feature is present in same order
# ✅ Works with model trained using crime_Assault / crime_Cyber_Crime etc.
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html
import joblib, os, math, requests, polyline
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="S.A.F.E. — AI Safe Route Planner", layout="wide", page_icon="🛰️")

# ----------------------------
# THEME (CYBERPUNK)
# ----------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at center, #030014 0%, #000000 100%);
    animation: pulse 10s infinite alternate;
    color: #E2E8F0;
    font-family: 'Poppins', sans-serif;
}
@keyframes pulse {
  0% { background: radial-gradient(circle at center, #030014 0%, #000000 100%); }
  50% { background: radial-gradient(circle at center, #050530 0%, #000000 100%); }
  100% { background: radial-gradient(circle at center, #030014 0%, #000000 100%); }
}
h1, h2, h3 { color: #00fff7; text-shadow: 0 0 15px #00fff7; }
.sidebar .sidebar-content {
    background: rgba(10, 20, 40, 0.8);
    border-right: 2px solid #00fff7;
}
.glass {
    background: rgba(15, 30, 50, 0.6);
    border: 1px solid rgba(0,255,247,0.3);
    border-radius: 16px;
    box-shadow: 0 0 25px rgba(0,255,247,0.2);
    padding: 1rem;
}
.metric { text-align:center; font-size:1.3rem; margin: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# UTILS
# ----------------------------
def haversine_m(p1, p2):
    R = 6371000.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def osrm_route(origin, destination, alternatives=False):
    o = f"{origin[1]},{origin[0]}"
    d = f"{destination[1]},{destination[0]}"
    url = f"http://router.project-osrm.org/route/v1/driving/{o};{d}"
    params = {"overview":"full","alternatives":str(alternatives).lower(),"geometries":"polyline"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    routes=[]
    for route in data.get("routes", []):
        coords=polyline.decode(route["geometry"])
        routes.append({"coords":coords,"distance":route["distance"],"duration":route["duration"]})
    return routes

# ----------------------------
# LOAD MODEL + DATA
# ----------------------------
MODEL_PATH="crime_hotspot_model_v8.pkl"
SCALER_PATH="crime_scaler_v8.pkl"
FEATURES_PATH="features_v8.txt"
DATA_PATH="ghaziabad_synthetic_crimes.csv"

use_ai=False
if all(os.path.exists(p) for p in [MODEL_PATH,SCALER_PATH,FEATURES_PATH,DATA_PATH]):
    model=joblib.load(MODEL_PATH)
    scaler=joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        features=[line.strip() for line in f if line.strip()]
    df=pd.read_csv(DATA_PATH)
    use_ai=True
else:
    st.error("❌ Missing model/scaler/features/data files!")

# ----------------------------
# BUILD AI RISK GRID
# ----------------------------
@st.cache_data
def build_risk():
    df['date_time']=pd.to_datetime(df['date_time'])
    df['hour']=df['date_time'].dt.hour
    df['month']=df['date_time'].dt.month
    df['weekday']=df['date_time'].dt.weekday
    df['is_weekend']=df['weekday'].isin([5,6]).astype(int)

    # Normalize crime types
    df['crime_type']=df['crime_type'].astype(str).str.strip().str.replace('_',' ').str.title()

    # Create dummy vars
    dummies=pd.get_dummies(df['crime_type'], prefix='crime')
    dummies.columns=[c.replace(' ','_') for c in dummies.columns]

    df2=pd.concat([df,dummies],axis=1)
    lat_bins=np.linspace(df.latitude.min(),df.latitude.max(),15)
    lon_bins=np.linspace(df.longitude.min(),df.longitude.max(),15)
    df2['lat_bin']=pd.cut(df.latitude,lat_bins,labels=False)
    df2['lon_bin']=pd.cut(df.longitude,lon_bins,labels=False)
    df2['cell']=df2['lat_bin'].astype(str)+"_"+df2['lon_bin'].astype(str)

    agg=df2.groupby(['cell',df2['date_time'].dt.date]).agg({
        'crime_type':'count','severity':'mean','hour':'mean',
        'month':'first','weekday':'first','is_weekend':'first',
        **{c:'sum' for c in dummies.columns}
    }).reset_index().rename(columns={'crime_type':'crime_count','hour':'avg_hour'})

    agg['lag1']=agg.groupby('cell')['crime_count'].shift(1).fillna(0)
    agg['lag2']=agg.groupby('cell')['crime_count'].shift(2).fillna(0)
    agg['rolling_3']=agg.groupby('cell')['crime_count'].rolling(3).mean().reset_index(level=0,drop=True).fillna(0)
    agg['month_sin']=np.sin(2*np.pi*agg['month']/12)
    agg['month_cos']=np.cos(2*np.pi*agg['month']/12)
    agg['weekday_sin']=np.sin(2*np.pi*agg['weekday']/7)
    agg['weekday_cos']=np.cos(2*np.pi*agg['weekday']/7)
    latest=agg.fillna(0).copy()

    # ✅ FIX: Force exact feature matching
    current_cols = list(latest.columns)
    rename_map = {}
    for col in current_cols:
        for f in features:
            if col.lower().replace('_','') == f.lower().replace('_',''):
                rename_map[col] = f
                break
    latest = latest.rename(columns=rename_map)

    # ✅ Fill any missing features with 0
    for f in features:
        if f not in latest.columns:
            latest[f]=0

    X = latest[features].fillna(0)
    Xs = scaler.transform(X)
    latest['pred_proba']=model.predict_proba(Xs)[:,1]

    cent=df2.groupby('cell')[['latitude','longitude']].mean().reset_index()
    grid=cent.merge(latest[['cell','pred_proba']],on='cell',how='right').fillna(0)
    return grid

if use_ai:
    cell_df=build_risk()
    kd=KDTree(cell_df[['latitude','longitude']],metric='euclidean')

# ----------------------------
# ROUTE SCORING
# ----------------------------
def score_route(coords,gender="any"):
    if not use_ai: return 0
    seg_risks,seg_lens=[],[]
    for i in range(len(coords)-1):
        a,b=coords[i],coords[i+1]
        mid=[(a[0]+b[0])/2,(a[1]+b[1])/2]
        _,idx=kd.query([mid],k=1)
        risk=float(cell_df.iloc[idx[0][0]]['pred_proba'])
        if gender=="female": risk*=1.25
        seg_risks.append(risk)
        seg_lens.append(haversine_m(a,b))
    return sum(r*l for r,l in zip(seg_risks,seg_lens))/sum(seg_lens)

# ----------------------------
# SIDEBAR CONFIG
# ----------------------------
st.sidebar.title("⚙️ Configuration")
origin_lat=st.sidebar.number_input("Origin Latitude", 28.6, 28.8, 28.67, 0.0001)
origin_lon=st.sidebar.number_input("Origin Longitude", 77.3, 77.6, 77.45, 0.0001)
dest_lat=st.sidebar.number_input("Destination Latitude", 28.6, 28.8, 28.70, 0.0001)
dest_lon=st.sidebar.number_input("Destination Longitude", 77.3, 77.6, 77.45, 0.0001)
gender=st.sidebar.selectbox("Gender",["any","female","male"],1)
alpha=st.sidebar.slider("Safety vs Time Tradeoff",0.0,1.0,0.8,0.05)
run_btn=st.sidebar.button("🚀 Generate AI Safe Route")

# ----------------------------
# MAIN UI
# ----------------------------
st.markdown("<h1 style='text-align:center;'>🛰️ S.A.F.E. — Smart AI For Everyone</h1>",unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI-Powered Safe Route Planner — Ghaziabad</h4>",unsafe_allow_html=True)
st.write(" ")

# ----------------------------
# GENERATE ROUTE
# ----------------------------
if run_btn:
    origin=(origin_lat,origin_lon)
    dest=(dest_lat,dest_lon)
    routes=osrm_route(origin,dest,alternatives=True)
    results=[]
    for r in routes:
        risk=score_route(r['coords'],gender)
        score=alpha*risk+(1-alpha)*(r['duration']/(r['duration']+60))
        results.append({"route":r,"risk":risk,"score":score})
    best=min(results,key=lambda x:x['score'])
    avg_risk=best['risk']
    dist=best['route']['distance']/1000
    dur=best['route']['duration']/60
    zone="High" if avg_risk>0.7 else "Medium" if avg_risk>0.4 else "Low"

    st.markdown("<div class='glass'>",unsafe_allow_html=True)
    colA,colB,colC=st.columns(3)
    colA.metric("🚗 Distance (km)",f"{dist:.2f}")
    colB.metric("🕒 Duration (min)",f"{dur:.1f}")
    colC.metric("🧠 AI Risk Index",f"{avg_risk*100:.1f}%")
    st.markdown(f"<div class='metric'>Predicted Hotspot Zone: <b style='color:#ff004c'>{zone}</b></div>",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
    st.write(" ")

    m=folium.Map(location=[(origin_lat+dest_lat)/2,(origin_lon+dest_lon)/2],zoom_start=12,tiles="CartoDB dark_matter")
    if use_ai:
        HeatMap(list(zip(cell_df.latitude,cell_df.longitude,cell_df.pred_proba)),radius=15,blur=20,min_opacity=0.4).add_to(m)
    folium.PolyLine(best['route']['coords'],weight=7,color="#00fff7",opacity=0.9).add_to(m)
    folium.CircleMarker(best['route']['coords'][0],radius=7,color="#00fff7",fill=True).add_to(m)
    folium.CircleMarker(best['route']['coords'][-1],radius=7,color="#ff004c",fill=True).add_to(m)
    html(m._repr_html_(),height=650)

    js="""<script>var msg=new SpeechSynthesisUtterance('AI Safe Route Generated Successfully');
    msg.pitch=1.1; msg.rate=1; msg.volume=1; speechSynthesis.speak(msg);</script>"""
    html(js)
else:
    st.info("Adjust parameters and click **Generate AI Safe Route**.")
