import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from utils import ensure_data, load_data
from train_model import train_and_save as train_supervised
from clustering import train_and_save_clusters

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction and Customer Segmentation System",
    layout="wide",
    page_icon="📊"
)

# ------------------------------------------------------------------
# Theme & animation CSS
# ------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
        .main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            text-align: center;
            background: linear-gradient(90deg, #ff4b4b, #2563eb, #ff4b4b);
            background-size: 200% auto;
            color: transparent;
            -webkit-background-clip: text;
            background-clip: text;
            animation: gradientMove 4s ease infinite;
        }
        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Customer Churn Prediction & Customer Segmentation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning powered telecom analytics dashboard</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Navigation buttons
# ------------------------------------------------------------------
PAGES = [
    "Home",
    "Models",
    "EDA Analysis",
    "Unsupervised Techniques",
    "Clustering",
    "ML Pipeline",
    "Source Code"
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

cols = st.columns(len(PAGES))
for i, page_name in enumerate(PAGES):
    if cols[i].button(page_name, use_container_width=True, key=f"nav_{page_name}"):
        st.session_state.page = page_name
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Ensure data & artifacts
# ------------------------------------------------------------------
data_path = ensure_data()
if not data_path:
    st.warning("Dataset not found and auto-download failed.")
    uploaded_fallback = st.file_uploader("Please upload the Telco Customer Churn CSV manually", type=["csv"])
    if uploaded_fallback is not None:
        data_path = ensure_data(uploaded_fallback)
    else:
        st.stop()

df = load_data(data_path)

if not os.path.exists("artifacts/supervised_metrics.csv"):
    with st.spinner("Training supervised models... please wait"):
        train_supervised(data_path)

if not os.path.exists("artifacts/cluster_scores.json"):
    with st.spinner("Training clustering models... please wait"):
        train_and_save_clusters(data_path)

@st.cache_resource
def load_artifacts():
    metrics_df = pd.read_csv("artifacts/supervised_metrics.csv")
    best_name = joblib.load("artifacts/best_model_name.joblib")
    best_pipe = joblib.load(f"artifacts/{best_name.replace(' ', '_')}_pipeline.joblib")
    cms = joblib.load("artifacts/confusion_matrices.joblib")

    with open("artifacts/cluster_scores.json") as f:
        cluster_scores = json.load(f)
    labels_dict = joblib.load("artifacts/cluster_labels.joblib")
    X_pca = joblib.load("artifacts/X_pca.joblib")
    with open("artifacts/cluster_summaries.json") as f:
        summaries = json.load(f)

    return metrics_df, best_name, best_pipe, cms, cluster_scores, labels_dict, X_pca, summaries

metrics_df, best_name, best_pipe, cms, cluster_scores, labels_dict, X_pca, summaries = load_artifacts()

# ------------------------------------------------------------------
# Page: Home
# ------------------------------------------------------------------
def home_page():
    st.header("🏠 Home - Churn Prediction")
    st.success(f"**Best supervised model selected:** {best_name}")

    uploaded = st.file_uploader("Upload a CSV file with customer features", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        if "TotalCharges" in input_df.columns:
            input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

        X = input_df.drop(["Churn", "customerID"], axis=1, errors="ignore")

        # Predict
        predictions = best_pipe.predict(X)
        probabilities = best_pipe.predict_proba(X)
        yes_idx = list(best_pipe.classes_).index("Yes")

        input_df["Churn Prediction"] = predictions
        input_df["Churn Probability"] = np.round(probabilities[:, yes_idx], 3)

        st.subheader("Prediction Results")
        st.dataframe(input_df.head(50), use_container_width=True)

        total = len(input_df)
        churn_count = int((predictions == "Yes").sum())
        churn_rate = churn_count / total * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", total)
        c2.metric("Predicted Churners", churn_count)
        c3.metric("Churn Rate", f"{churn_rate:.2f}%")

        fig = px.pie(
            names=["No Churn", "Churn"],
            values=[total - churn_count, churn_count],
            title="Predicted Churn Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        if "Churn" in input_df.columns:
            y_true = input_df["Churn"]
            acc = accuracy_score(y_true, predictions)
            prec = precision_score(y_true, predictions, pos_label="Yes", zero_division=0)
            rec = recall_score(y_true, predictions, pos_label="Yes", zero_division=0)
            f1 = f1_score(y_true, predictions, pos_label="Yes", zero_division=0)

            st.subheader("Evaluation on Uploaded Data")
            st.write({
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4)
            })

            cm = confusion_matrix(y_true, predictions, labels=["No", "Yes"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["No", "Yes"],
                y=["No", "Yes"],
                labels=dict(x="Predicted", y="Actual"),
                title="Confusion Matrix",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")

# ------------------------------------------------------------------
# Page: Models
# ------------------------------------------------------------------
def models_page():
    st.header("🤖 Supervised Models Comparison")
    st.write("Comparison of Decision Tree, Random Forest, KNN, Logistic Regression, and Naive Bayes.")

    styled = metrics_df.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-Score"],
        color="lightgreen"
    )
    st.dataframe(styled, use_container_width=True)

    fig = px.bar(
        metrics_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Model Performance Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_radar = go.Figure()
    for _, row in metrics_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Accuracy"], row["Precision"], row["Recall"], row["F1-Score"]],
            theta=["Accuracy", "Precision", "Recall", "F1-Score"],
            fill="toself",
            name=row["Model"]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Confusion Matrices")
    for name, cm in cms.items():
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["No", "Yes"],
            y=["No", "Yes"],
            title=f"{name} Confusion Matrix",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page: EDA
# ------------------------------------------------------------------
def eda_page():
    st.header("📊 Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.subheader("Dataset Sample")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Churn Distribution")
    churn_counts = df["Churn"].value_counts()
    fig = px.pie(values=churn_counts.values, names=churn_counts.index, hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn by Contract Type")
    contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    fig = px.bar(contract, barmode="group", title="Churn % by Contract")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tenure vs Churn")
    fig = px.box(df, x="Churn", y="tenure", color="Churn")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Charges vs Churn")
    fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Payment Method")
    payment = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
    fig = px.bar(payment, barmode="group", title="Churn % by Payment Method")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page: Unsupervised Techniques
# ------------------------------------------------------------------
def unsupervised_page():
    st.header("🧩 Unsupervised Learning - Customer Segmentation")

    st.markdown("""
    Three clustering algorithms are applied after scaling/encoding features and reducing them to 2 dimensions with PCA:
    - **K-Means Clustering**
    - **Hierarchical Clustering**
    - **DBSCAN**
    """)

    scores_df = pd.DataFrame([
        {
            "Algorithm": k,
            "Silhouette Score": v["silhouette"],
            "Clusters Found": v["n_clusters"]
        }
        for k, v in cluster_scores.items()
    ])
    st.dataframe(scores_df, use_container_width=True)

    fig = px.bar(
        scores_df,
        x="Algorithm",
        y="Silhouette Score",
        color="Algorithm",
        title="Silhouette Score Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    pca = joblib.load("artifacts/pca.joblib")
    ev = pca.explained_variance_ratio_
    fig = px.bar(
        x=["PC1", "PC2"],
        y=ev,
        labels={"x": "Component", "y": "Explained Variance Ratio"},
        title="PCA Explained Variance"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Visualizations on PCA Components")
    for name, labels in labels_dict.items():
        vis_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels.astype(str)
        })
        fig = px.scatter(
            vis_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=f"{name} Clustering",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page: Clustering
# ------------------------------------------------------------------
def clustering_page():
    st.header("🎯 Clustering Results & Group Behaviour")

    algo = st.selectbox("Select Clustering Algorithm", list(labels_dict.keys()))
    labels = labels_dict[algo]

    df_c = df.copy()
    df_c["Cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    st.info(f"Number of clusters (excluding noise): {n_clusters}")

    # Cluster size distribution
    dist = pd.Series(labels).value_counts().sort_index()
    fig = px.bar(
        x=dist.index.astype(str),
        y=dist.values,
        labels={"x": "Cluster", "y": "Count"},
        title="Cluster Size Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary
    summary_df = pd.DataFrame(summaries[algo])
    st.subheader("Cluster Behaviour Summary")
    st.dataframe(
        summary_df.style.format({
            "Avg_Tenure": "{:.2f}",
            "Avg_MonthlyCharges": "${:.2f}",
            "Avg_TotalCharges": "${:.2f}",
            "Churn_Rate": "{:.2%}"
        }),
        use_container_width=True
    )

    # Scatter
    st.subheader("Customer Segments: Tenure vs Monthly Charges")
    fig = px.scatter(
        df_c,
        x="tenure",
        y="MonthlyCharges",
        color=df_c["Cluster"].astype(str),
        title="Customer Segments",
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Business Insights")
    st.markdown("""
    - **High-risk segment:** Low tenure + high monthly charges → offer discounts or loyalty programs.
    - **Loyal low-revenue segment:** High tenure + low charges → upsell premium services.
    - **Stable segment:** High tenure + high charges → priority customer retention.
    - Use targeted marketing campaigns based on each cluster's behavior.
    """)

# ------------------------------------------------------------------
# Page: ML Pipeline
# ------------------------------------------------------------------
def pipeline_page():
    st.header("⚙️ Machine Learning Pipeline")

    steps = [
        "1. **Data Collection** - Load Telco Customer Churn dataset.",
        "2. **Data Cleaning** - Convert `TotalCharges` to numeric, handle missing values, drop `customerID`.",
        "3. **Preprocessing** - Scale numeric features; one-hot encode categorical features.",
        "4. **Train/Test Split** - 80/20 stratified split.",
        "5. **Supervised Learning** - Train and evaluate 5 classification models.",
        "6. **Model Selection** - Pick the best model by F1-Score for churn prediction.",
        "7. **Unsupervised Learning** - PCA + K-Means, Hierarchical, DBSCAN for segmentation.",
        "8. **Evaluation** - Compare clusters with silhouette scores and behavior summaries.",
        "9. **Deployment** - Streamlit interactive dashboard."
    ]

    for step in steps:
        st.markdown(f"- {step}")

    st.subheader("Preprocessing Code Snippet")
    code = '''
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])
'''
    st.code(code, language="python")

# ------------------------------------------------------------------
# Page: Source Code
# ------------------------------------------------------------------
def source_page():
    st.header("📝 Source Code")

    files = {
        "app.py": "app.py",
        "train_model.py": "train_model.py",
        "clustering.py": "clustering.py",
        "utils.py": "utils.py",
        "requirements.txt": "requirements.txt"
    }

    for label, path in files.items():
        with st.expander(f"📄 {label}"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lang = "python" if path.endswith(".py") else "text"
                    st.code(f.read(), language=lang)
            except Exception as e:
                st.error(f"Could not read {path}: {e}")

# ------------------------------------------------------------------
# Route to selected page
# ------------------------------------------------------------------
page = st.session_state.page

if page == "Home":
    home_page()
elif page == "Models":
    models_page()
elif page == "EDA Analysis":
    eda_page()
elif page == "Unsupervised Techniques":
    unsupervised_page()
elif page == "Clustering":
    clustering_page()
elif page == "ML Pipeline":
    pipeline_page()
elif page == "Source Code":
    source_page()

st.divider()
st.markdown(
    "<center><small>Built with ❤️ using Python & Streamlit | Open Ended Lab - Machine Learning</small></center>",
    unsafe_allow_html=True
)