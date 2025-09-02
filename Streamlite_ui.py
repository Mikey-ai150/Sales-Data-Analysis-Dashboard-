import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import plotly.express as px

# Title
st.set_page_config(layout="wide")
st.title("📊 Smart Dataset Analyzer & Model Evaluator")

# File upload
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(uploaded_file, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"❌ Failed to read the file: {e}")
    else:
        st.success("✅ File uploaded successfully!")
        st.subheader("📌 Data Preview")
        st.dataframe(df.head())

        # Show basic info
        st.subheader("🔎 Dataset Summary")
        st.write(df.describe())

        st.subheader("🧼 Missing Values")
        st.write(df.isnull().sum())

        # EDA Section
        st.subheader("📊 Data Distribution")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_eda_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
            if selected_eda_col in df.columns:
                fig1 = plt.figure()
                sns.histplot(df[selected_eda_col], kde=True)
                st.pyplot(fig1)
            else:
                st.warning("⚠️ Selected column not found.")
        else:
            st.warning("⚠️ No numeric columns available to visualize.")

        # Correlation Heatmap
        if len(numeric_cols) >= 2:
            st.subheader("📌 Correlation Heatmap")
            fig2 = plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig2)

        # Classification / Regression Section
        st.subheader("🎯 Model Training")
        target_col = st.selectbox("Select Target Column", df.columns)
        feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])

        # Remove datetime columns from feature set
        feature_cols = [col for col in feature_cols if not np.issubdtype(df[col].dtype, np.datetime64)]

        if st.button("Train Model"):
            try:
                X = df[feature_cols]
                y = df[target_col]

                # Drop rows with missing target or features
                data = pd.concat([X, y], axis=1).dropna()
                X = data[feature_cols]
                y = data[target_col]

                # Encode categorical features
                X = pd.get_dummies(X, drop_first=True)

                # Determine task type
                if pd.api.types.is_numeric_dtype(y):
                    if y.nunique() <= 10:
                        task = "classification"
                    else:
                        task = "regression"
                else:
                    y = y.astype(str)
                    task = "classification"

                st.info(f"Detected task type: **{task.upper()}**")

                # Encode target if classification
                if task == "classification":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()

                # Split and train
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Show results
                if task == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Classification Accuracy: {acc:.2f}")
                    st.subheader("📋 Classification Report")
                    st.text(classification_report(y_test, y_pred))
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.success(f"✅ Regression Results: R² = {r2:.2f}, RMSE = {np.sqrt(mse):.2f}")

                # Feature importance
                importances = pd.Series(model.feature_importances_, index=X.columns)
                top_importances = importances.sort_values(ascending=False).head(10)
                st.subheader("📈 Top 10 Important Features")
                fig3 = plt.figure()
                top_importances.plot(kind='barh', color='skyblue')
                plt.gca().invert_yaxis()
                st.pyplot(fig3)

            except Exception as e:
                st.error(f"❌ Error during model training: {e}")
