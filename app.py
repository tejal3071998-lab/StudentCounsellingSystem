import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Page Configuration
st.set_page_config(
    page_title="Student Counselling System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.title("🎓 Student Career Counselling & Recommendation System")
# st.markdown("""
# This system uses **Machine Learning** to assist students in identifying their career path.
# * **Linear Regression** is used to forecast the student's **Career Readiness Score**.
# * **Logistic Regression** is used to predict the most suitable **Recommended Course**.
# ---
# """)


# Load Dataset & Cache it
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("student_counselling_dataset_1000_realistic.csv")
        # Fill missing values for 10th graders who don't have a previous stream
        df['Previous_Stream'] = df['Previous_Stream'].fillna('None')
        return df
    except FileNotFoundError:
        st.error(
            "❌ Dataset file 'student_counselling_dataset_1000_realistic.csv' not found! Please place it in the same directory.")
        return None


df = load_data()

if df is not None:
    # ------------------ ML Model Training Engine (Cached) ------------------
    @st.cache_resource
    def train_models(data):
        # Feature columns
        categorical_cols = ['Class', 'Previous_Stream', 'Primary_Skill', 'Interest', 'Budget']
        numerical_cols = ['Percentage']

        X = data[categorical_cols + numerical_cols]
        y_reg = data['Career_Readiness_Score']
        y_clf = data['Recommended_Course']

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        # 1. Linear Regression Pipeline
        lin_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # 2. Logistic Regression Pipeline
        log_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        # Train-Test Splits for metrics evaluation
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

        # Fit models
        lin_pipeline.fit(X_train, y_train_reg)
        log_pipeline.fit(X_train, y_train_clf)

        # Calculate Metrics
        lin_pred = lin_pipeline.predict(X_test)
        log_pred = log_pipeline.predict(X_test)

        metrics = {
            "lin_r2": r2_score(y_test_reg, lin_pred),
            "lin_mae": mean_absolute_error(y_test_reg, lin_pred),
            "log_acc": accuracy_score(y_test_clf, log_pred)
        }

        return lin_pipeline, log_pipeline, metrics


    # Train models on launch
    lin_model, log_model, model_metrics = train_models(df)

    # ------------------ Navigation Sidebar ------------------
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Counselling Panel", "Dataset Dashboard", "Model Performance"])

    # ------------------ MODE 1: COUNSELLING PANEL ------------------
    if app_mode == "Counselling Panel":
        st.header("🔍 Interactive Student Assessment")
        st.write("Fill out the profile details below to discover personalized recommendations.")

        col1, col2 = st.columns(2)

        with col1:
            student_class = st.selectbox("Current Class Level", options=list(df['Class'].unique()))

            # Dynamic control: if 10th grade, previous stream is automatically "None"
            if student_class == "10th":
                prev_stream = "None"
                st.info("Previous Stream defaulted to 'None' for 10th Grade.")
            else:
                prev_stream = st.selectbox("Previous Stream (12th Grade Only)",
                                           options=[opt for opt in df['Previous_Stream'].unique() if opt != "None"])

            # 📥 Swapped from slider to an explicit numeric input text box
            percentage = st.number_input(
                "Aggregate Academic Percentage (%)",
                min_value=0,
                max_value=100,
                value=75,
                step=1,
                help="Enter a value between 0 and 100"
            )

        with col2:
            primary_skill = st.selectbox("Primary Core Skill", options=list(df['Primary_Skill'].unique()))
            interest_area = st.selectbox("Primary Field of Interest", options=list(df['Interest'].unique()))
            budget_level = st.selectbox("Financial Budget Level", options=list(df['Budget'].unique()))

        # Execution Button
        st.markdown("---")
        if st.button("🚀 Generate Personalised Counselling Report", use_container_width=True):
            # Construct DataFrame input format matching original
            input_data = pd.DataFrame([{
                'Class': student_class,
                'Previous_Stream': prev_stream,
                'Percentage': percentage,
                'Primary_Skill': primary_skill,
                'Interest': interest_area,
                'Budget': budget_level
            }])

            # Predict
            predicted_score = lin_model.predict(input_data)[0]
            predicted_course = log_model.predict(input_data)[0]

            # Bound readiness score within reasonable bounds [0-100]
            predicted_score = max(0, min(100, predicted_score))

            # Display Results
            st.success("🎉 Recommendations successfully compiled!")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(
                    label="⚡ Forecasted Career Readiness Score",
                    value=f"{predicted_score:.1f} / 100"
                )
                st.progress(int(predicted_score))
                st.caption("This rating evaluates your readiness metrics against target professional ecosystems.")

            with res_col2:
                st.metric(
                    label="🎓 Best Fit Recommended Course",
                    value=str(predicted_course)
                )
                st.info(
                    f"Based on your profile, pursuing **{predicted_course}** coordinates ideally with your interest in *{interest_area}*.")

    # ------------------ MODE 2: DATASET DASHBOARD ------------------
    elif app_mode == "Dataset Dashboard":
        st.header("📊 Student Base Historical Insights")

        # Basic Metrics Display
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Profiles Analyzed", len(df))
        m2.metric("Average Student Score %", f"{df['Percentage'].mean():.1f}%")
        m3.metric("Unique Recommended Courses", len(df['Recommended_Course'].unique()))

        st.markdown("### Sample Historical Profiles")
        st.dataframe(df.head(10), use_container_width=True)

        # Exploratory Data Visualizations
        st.markdown("### Data Core Relations")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='Percentage', y='Career_Readiness_Score', hue='Class', ax=ax, palette='Set2')
            ax.set_title("Percentage vs Career Readiness Score")
            st.pyplot(fig)

        with v_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='Interest', order=df['Interest'].value_counts().index, ax=ax, palette='viridis')
            ax.set_title("Distribution of Student Interests")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ------------------ MODE 3: MODEL PERFORMANCE ------------------
    elif app_mode == "Model Performance":
        st.header("⚙️ Machine Learning Model Breakdown")
        st.write("Below are validation stats from the background training split (80% Train, 20% Test).")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Linear Regression Model")
            st.markdown("**Target**: `Career_Readiness_Score`")
            st.metric("R² Score (Accuracy Variance)", f"{model_metrics['lin_r2']:.4f}")
            st.metric("Mean Absolute Error (MAE)", f"{model_metrics['lin_mae']:.2f}")
            st.caption(
                "An R² close to 1 means the system accurately reads factors affecting career performance scores.")

        with c2:
            st.subheader("Logistic Regression Model")
            st.markdown("**Target**: `Recommended_Course`")
            st.metric("Categorical Testing Accuracy", f"{model_metrics['log_acc'] * 100:.2f}%")
            st.caption("Indicates the exact prediction match rate over validation test subsets.")