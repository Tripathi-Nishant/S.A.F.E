# 🛰️ S.A.F.E. — Smart AI For Everyone  
# 🚦 AI-Powered Crime Prediction & Safe Route Planner

> Empowering citizens with **AI-driven safety intelligence** — real-time hotspot detection and safe route generation for smarter, safer cities.

---

# 🌍 Project Overview

**S.A.F.E. (Smart AI For Everyone)** is a next-generation web application that predicts **crime-prone zones** and helps users choose **the safest travel routes** using Artificial Intelligence.  
It combines **Machine Learning (XGBoost)**, **spatial data processing**, and **interactive mapping** to visualize risks and optimize safety in urban navigation.

Built for **Ghaziabad**, scalable to **any smart city** across the world 🌐.

---

## 💡 Key Features

✅ **AI Crime Hotspot Prediction** — Predicts future high-risk zones using historical data  
✅ **Interactive Heatmap** — Visualizes danger probability on an intuitive city map  
✅ **Safe Route Suggestion** — AI calculates safest paths using OSRM routing  
✅ **AI Voice Feedback** — Announces when a safe route is generated  
✅ **Neon Cyber UI** — Futuristic glowing interface with pulse animation  
✅ **Modular Architecture** — Fully ready for city-wide expansion  

---

## 🧠 Tech Stack

| Layer | Technologies |
|--------|---------------|
| **Frontend** | Streamlit, Folium, HTML/CSS (custom cyber theme) |
| **Backend** | Python, Pandas, NumPy, Scikit-learn |
| **Machine Learning** | XGBoost, StandardScaler |
| **Visualization** | Folium Heatmaps, Matplotlib |
| **Routing Engine** | OSRM (Open Source Routing Machine) |
| **Deployment** | Streamlit Cloud / Render / Hugging Face Spaces |

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Tripathi-Nishant/S.A.F.E.git
cd S.A.F.E

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run safe_route_app_v8.py
