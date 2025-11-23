import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="💎 Diamond Dynamics", layout="centered")

st.title("💎 Diamond Dynamics: Price Prediction & Market Segmentation")

# Load models
@st.cache_resource
def load_regressor():
    with open(r"C:\Users\user\Documents\Guvi\models\best_regressor.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["feature_cols"]

@st.cache_resource
def load_clusterer():
    with open(r"C:\Users\user\Documents\Guvi\models\best_cluster.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["feature_cols"], data["cluster_names"]


reg_model, reg_scaler, reg_features = load_regressor()
cluster_model, cluster_scaler, cluster_features, cluster_names = load_clusterer()

st.sidebar.header("Diamond Attributes")

# User inputs
carat = st.sidebar.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
depth = st.sidebar.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
table = st.sidebar.number_input("Table (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
x = st.sidebar.number_input("Length x (mm)", min_value=3.0, max_value=12.0, value=6.0, step=0.01)
y = st.sidebar.number_input("Width y (mm)", min_value=3.0, max_value=12.0, value=6.0, step=0.01)
z = st.sidebar.number_input("Depth z (mm)", min_value=2.0, max_value=8.0, value=3.8, step=0.01)

cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# Ordinal encoding (must match training)
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

cut_enc = cut_order.index(cut)
color_enc = color_order.index(color)
clarity_enc = clarity_order.index(clarity)

# Feature engineering like training
volume = x * y * z
dim_ratio = (x + y) / (2 * z)

# For regression: we also used price_per_carat during training, which depends on price.
# During prediction we don't know price, so we can exclude that feature in training OR keep it simple:
# Option chosen: we trained with price_per_carat; for app, we re-train without that column.
# (So make sure training script uses same reg_features list as here.)

# Build feature vector
feature_dict = {
    "carat": carat,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
    "volume": volume,
    "dim_ratio": dim_ratio,
    "cut_enc": cut_enc,
    "color_enc": color_enc,
    "clarity_enc": clarity_enc,
}

# Ensure order matches training
X_input = np.array([[feature_dict[col] for col in reg_features]])

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("💰 Predict Price"):
        X_scaled = reg_scaler.transform(X_input)
        price_inr_pred = reg_model.predict(X_scaled)[0]
        st.subheader(f"Estimated Price: ₹ {price_inr_pred:,.0f}")

with col2:
    if st.button("📊 Predict Market Segment"):
        X_clust_input = np.array([[feature_dict[col] for col in cluster_features]])
        X_clust_scaled = cluster_scaler.transform(X_clust_input)
        cluster_label = cluster_model.predict(X_clust_scaled)[0]
        cluster_name = cluster_names.get(cluster_label, f"Cluster {cluster_label}")
        st.subheader(f"Market Segment: {cluster_name}")
        st.caption(f"(Cluster ID: {cluster_label})")
