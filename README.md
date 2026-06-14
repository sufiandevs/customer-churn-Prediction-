
# Customer Churn Prediction and Customer Segmentation System

A complete Streamlit-based ML dashboard for telecom customer churn prediction and customer segmentation using supervised and unsupervised learning.

## Dataset

- **Kaggle Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- The app auto-downloads the dataset from a direct raw URL. If the download fails, upload the CSV manually.

## Features

- **Home:** Upload any customer CSV and get churn predictions + probabilities.
- **Models:** Compare Decision Tree, Random Forest, KNN, Logistic Regression, Naive Bayes with metrics & charts.
- **EDA Analysis:** Explore customer behavior patterns with interactive visualizations.
- **Unsupervised Techniques:** Compare K-Means, Hierarchical, and DBSCAN clustering.
- **Clustering:** Analyze cluster behavior and business insights.
- **ML Pipeline:** View the full preprocessing & modeling pipeline.
- **Source Code:** View all project source files directly in the app.

## Local Setup

```bash
git clone https://github.com/yourusername/churn_app.git
cd churn_app
pip install -r requirements.txt
streamlit run app.py
