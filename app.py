import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️"
)

st.title("🛍️ Customer Segmentation using K-Means")

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Remove missing values
df = df.dropna()

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset Statistics
st.subheader("Dataset Statistics")
st.write(df.describe())

# Features for Clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# K-Means Model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# Cluster Counts
st.subheader("Customers per Cluster")
st.bar_chart(
    df["Cluster"].value_counts().sort_index()
)

# Scatter Plot
st.subheader("Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(8,5))

scatter = ax.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segmentation using K-Means")

st.pyplot(fig)

# Cluster Selection
cluster_no = st.selectbox(
    "Select Cluster",
    sorted(df["Cluster"].unique())
)

st.subheader(f"Customers in Cluster {cluster_no}")

st.dataframe(
    df[df["Cluster"] == cluster_no]
)

st.success("K-Means Clustering Applied Successfully")

st.markdown("---")
st.caption("Customer Segmentation | Unsupervised Learning Project")