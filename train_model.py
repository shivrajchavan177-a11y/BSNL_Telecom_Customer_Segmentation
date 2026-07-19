import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ================================
# Create folders
# ================================
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ================================
# Load Dataset
# ================================
df = pd.read_csv("Data/clean/Cleaned_BSNL_Data.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ================================
# Numerical & Categorical Columns
# ================================
numerical_cols = [
    "SeniorCitizen",
    "Tenure_Months",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_cols = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "StreamingTV",
    "StreamingMovies",
    "TechSupport",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "MultipleLines"
]

# ================================
# Preprocessing
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ================================
# Transform Data
# ================================
X = preprocessor.fit_transform(df)

# ================================
# Train KMeans
# ================================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# ================================
# Calculate Silhouette Score
# ================================
sil_score = silhouette_score(X, clusters)

print(f"Silhouette Score: {sil_score:.3f}")

# ================================
# Add Cluster Labels
# ================================
df["Cluster"] = clusters

# ================================
# Business Names
# ================================
cluster_map = {
    0: "👑 Loyal High-Value Customers",
    1: "🆕 New Customers",
    2: "⚠️ At-Risk Customers",
    3: "💎 Premium Service Users"
}

df["Customer Segment"] = df["Cluster"].map(cluster_map)

# ================================
# Save Files
# ================================
joblib.dump(preprocessor, "model/preprocessor.pkl")
joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(sil_score, "model/silhouette_score.pkl")
    
df.to_csv("data/Segmented_BSNL_Customers.csv", index=False)

print("\nTraining Completed Successfully!\n")

print(df["Customer Segment"].value_counts())

print("\nFiles Saved:")
print("✔ model/preprocessor.pkl")
print("✔ model/kmeans_model.pkl")
print("✔ model/silhouette_score.pkl")
print("✔ data/Segmented_BSNL_Customers.csv")