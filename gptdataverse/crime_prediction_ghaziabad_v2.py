# =======================================================
# Crime Hotspot Prediction Model - Ghaziabad (v8 Ultra-Stable)
# =======================================================
# 🚀 Highlights:
# ✅ Fixes "Cannot sample ... when replace=False" crash
# ✅ Fixes feature name mismatch (case + spacing)
# ✅ Automatically exports features_v8.txt for Streamlit
# ✅ Handles imbalance & feature scaling correctly
# ✅ Produces hotspot heatmap + feature importance chart
# =======================================================

# pip install pandas numpy xgboost scikit-learn folium matplotlib joblib

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample, compute_sample_weight
from xgboost import XGBClassifier
import folium
from folium.plugins import HeatMap
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "ghaziabad_synthetic_crimes.csv"
UNDERSAMPLE_IF_RATIO_GT = 8
RANDOM_STATE = 42

# -----------------------------
# 1. Load dataset
# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Total records: {len(df)}")
print("Columns:", df.columns.tolist())

# -----------------------------
# 2. Preprocessing
# -----------------------------
print("\n🧹 Preprocessing data...")
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# ✅ Normalize crime_type naming
df['crime_type'] = (
    df['crime_type']
    .astype(str)
    .str.strip()
    .str.replace('_', ' ')
    .str.title()
)

# One-hot encode crime type
crime_dummies = pd.get_dummies(df['crime_type'], prefix="crime")
# Normalize dummy names (no spaces)
crime_dummies.columns = [c.strip().replace(" ", "_") for c in crime_dummies.columns]
df = pd.concat([df, crime_dummies], axis=1)

# -----------------------------
# 3. Grid creation and aggregation
# -----------------------------
def create_grid_and_aggregate(df, lat_bins_count=15, lon_bins_count=15):
    lat_bins = np.linspace(df.latitude.min(), df.latitude.max(), lat_bins_count)
    lon_bins = np.linspace(df.longitude.min(), df.longitude.max(), lon_bins_count)
    df['lat_bin'] = pd.cut(df.latitude, lat_bins, labels=False)
    df['lon_bin'] = pd.cut(df.longitude, lon_bins, labels=False)
    df['cell'] = df['lat_bin'].astype(str) + "_" + df['lon_bin'].astype(str)

    agg = df.groupby(['cell', df['date_time'].dt.date]).agg({
        'crime_type': 'count',
        'severity': 'mean',
        'hour': 'mean',
        'month': 'first',
        'weekday': 'first',
        'is_weekend': 'first',
        **{col: 'sum' for col in crime_dummies.columns}
    }).reset_index()

    agg = agg.rename(columns={'crime_type': 'crime_count', 'hour': 'avg_hour'})
    agg['date'] = pd.to_datetime(agg['date_time'])
    agg = agg.sort_values(['cell', 'date'])
    agg['next_day_crime'] = agg.groupby('cell')['crime_count'].shift(-1)
    agg['target'] = (agg['next_day_crime'] > 0).astype(int)
    return agg.dropna(subset=['target'])

print("\n🧩 Creating 15x15 grid...")
agg = create_grid_and_aggregate(df, 15, 15)
if len(agg['target'].unique()) < 2:
    print("⚠️ Only one class found! Switching to 10x10 grid...")
    agg = create_grid_and_aggregate(df, 10, 10)

print("Target distribution:\n", agg['target'].value_counts())

# -----------------------------
# 4. Feature Engineering
# -----------------------------
print("\n⚙️ Creating features...")
agg['lag1'] = agg.groupby('cell')['crime_count'].shift(1).fillna(0)
agg['lag2'] = agg.groupby('cell')['crime_count'].shift(2).fillna(0)
agg['rolling_3'] = (
    agg.groupby('cell')['crime_count']
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
    .fillna(0)
)

# Temporal features
agg['month_sin'] = np.sin(2 * np.pi * agg['month'] / 12)
agg['month_cos'] = np.cos(2 * np.pi * agg['month'] / 12)
agg['weekday_sin'] = np.sin(2 * np.pi * agg['weekday'] / 7)
agg['weekday_cos'] = np.cos(2 * np.pi * agg['weekday'] / 7)

crime_features = [c for c in agg.columns if c.startswith("crime_")]
features = [
    'crime_count','severity','avg_hour',
    'lag1','lag2','rolling_3',
    'month_sin','month_cos','weekday_sin','weekday_cos','is_weekend'
] + crime_features

X = agg[features].fillna(0)
y = agg['target']

# -----------------------------
# 5. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
)
print("\n📊 Train class distribution:")
print(y_train.value_counts())

# -----------------------------
# 5B. Handle Imbalance (SAFE FIX)
# -----------------------------
ratio = y_train.value_counts().max() / y_train.value_counts().min()
if ratio > UNDERSAMPLE_IF_RATIO_GT:
    print(f"⚠️ Severe imbalance detected ({ratio:.2f}:1). Applying undersampling...")
    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    maj = train_df[train_df['target'] == 1]  # majority class (1 = hotspot next day)
    minc = train_df[train_df['target'] == 0]  # minority class

    # ✅ Safe undersample size
    target_samples = min(len(maj), max(len(minc) * 3, len(minc)))
    print(f"Undersampling majority class from {len(maj)} → {target_samples}")

    maj_down = resample(
        maj,
        replace=False,
        n_samples=target_samples,
        random_state=RANDOM_STATE,
    )
    train_bal = pd.concat([maj_down, minc]).sample(frac=1, random_state=RANDOM_STATE)
    print("Post-undersample counts:", train_bal['target'].value_counts().to_dict())

    X_train = train_bal[features]
    y_train = train_bal['target']
else:
    print("⚖️ Imbalance within acceptable range.")

# -----------------------------
# 6. Align & Scale Features
# -----------------------------
print("\n🔄 Aligning feature names and scaling data...")

features = list(dict.fromkeys(features))
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

X_train = X_train.reindex(columns=features, fill_value=0)
X_test = X_test.reindex(columns=features, fill_value=0)

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train)
X_test_np = scaler.transform(X_test)
print(f"✅ Scaling complete — {len(features)} features aligned")

# -----------------------------
# 7. Train Model (XGBoost)
# -----------------------------
print("\n🚀 Training XGBoost model...")
sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1,
    reg_alpha=0.2,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
)
model.fit(X_train_np, y_train, sample_weight=sample_weight)

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = model.predict(X_test_np)
print("\n===== 🧾 Model Evaluation =====")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# -----------------------------
# 9. Save Model + Scaler + Features
# -----------------------------
joblib.dump(model, "crime_hotspot_model_v8.pkl")
joblib.dump(scaler, "crime_scaler_v8.pkl")
with open("features_v8.txt", "w") as f:
    f.write("\n".join(features))
print("\n💾 Saved: crime_hotspot_model_v8.pkl, crime_scaler_v8.pkl, features_v8.txt")

# -----------------------------
# 10. Generate Heatmap
# -----------------------------
print("\n🗺️ Generating hotspot heatmap...")
agg_pred = agg.copy()
X_all = agg_pred[features].reindex(columns=features, fill_value=0)
X_all = scaler.transform(X_all)
agg_pred['pred_proba'] = model.predict_proba(X_all)[:, 1]

risk = agg_pred.groupby('cell')['pred_proba'].mean().reset_index()
centroids = df.groupby('cell')[['latitude', 'longitude']].mean().reset_index()
heat_data = risk.merge(centroids, on='cell')

m = folium.Map(location=[28.67, 77.45], zoom_start=12, tiles="CartoDB dark_matter")
HeatMap(
    list(zip(heat_data.latitude, heat_data.longitude, heat_data.pred_proba)),
    radius=12, blur=20, max_zoom=1, min_opacity=0.4
).add_to(m)
m.save("ghaziabad_predicted_hotspots_v8.html")
print("✅ Saved interactive heatmap: ghaziabad_predicted_hotspots_v8.html")

# -----------------------------
# 11. Feature Importance Plot
# -----------------------------
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(15), importances[sorted_idx][:15])
plt.xticks(range(15), np.array(features)[sorted_idx][:15], rotation=45, ha='right')
plt.title("Top 15 Feature Importances - Crime Prediction (v8)")
plt.tight_layout()
plt.savefig("feature_importance_v8.png")
print("📊 Saved feature importance chart: feature_importance_v8.png")

print("\n✅ All done! Model trained successfully, safe and consistent with Streamlit app v8.")
