import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

# Page Configuration
st.set_page_config(
    page_title="Student Counselling System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed by default as sidebar isn't needed
)

# App Title & Description
st.title("🎓 Student Career Counselling & Recommendation System")



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

        # Fit models on full dataset to maximize training pattern learning
        lin_pipeline.fit(X, y_reg)
        log_pipeline.fit(X, y_clf)

        return lin_pipeline, log_pipeline


    # Train models on launch
    lin_model, log_model = train_models(df)


 # Course Recommendation Dictionary
    course_recommendations = {

        "Engineering": {
            "best": "Bachelor of Technology (B.Tech)",
            "others": [
                "B.Tech AI & Machine Learning",
                "B.Tech Computer Science",
                "B.Tech Information Technology",
                "BCA",
                "B.Sc Computer Science"
            ]
        },

        "Medical": {
            "best": "MBBS",
            "others": [
                "BDS",
                "BAMS",
                "BHMS",
                "B.Sc Nursing",
                "Bachelor of Pharmacy"
            ]
        },

        "Commerce": {
            "best": "Bachelor of Commerce (B.Com)",
            "others": [
                "Bachelor of Business Administration (BBA)",
                "Chartered Accountant (CA)",
                "Company Secretary (CS)",
                "Bachelor of Economics",
                "Bachelor of Accounting & Finance"
            ]
        },

        "Arts": {
            "best": "Bachelor of Arts (BA)",
            "others": [
                "Bachelor of Journalism",
                "Bachelor of Fine Arts",
                "Bachelor of Social Work",
                "Bachelor of Psychology",
                "Bachelor of Laws (LLB)"
            ]
        },

        "Computer Science": {
            "best": "Bachelor of Computer Applications (BCA)",
            "others": [
                "B.Tech Computer Science",
                "B.Sc Computer Science",
                "B.Tech Artificial Intelligence",
                "B.Tech Cyber Security",
                "B.Sc Information Technology"
            ]
        },

        "Business": {
            "best": "Bachelor of Business Administration (BBA)",
            "others": [
                "B.Com",
                "Bachelor of Management Studies",
                "Bachelor of Economics",
                "Digital Marketing",
                "MBA (After Graduation)"
            ]
        },

        "Law": {
            "best": "Bachelor of Laws (LLB)",
            "others": [
                "BA LLB",
                "BBA LLB",
                "B.Com LLB",
                "Corporate Law",
                "Cyber Law"
            ]
        },

        "Design": {
            "best": "Bachelor of Design (B.Des)",
            "others": [
                "Fashion Designing",
                "Interior Designing",
                "Graphic Designing",
                "Animation & VFX",
                "UI/UX Design"
            ]
        },

        "Agriculture": {
            "best": "B.Sc Agriculture",
            "others": [
                "Horticulture",
                "Food Technology",
                "Forestry",
                "Veterinary Science",
                "Agricultural Engineering"
            ]
        },

        "Hospitality": {
            "best": "Bachelor of Hotel Management (BHM)",
            "others": [
                "Travel & Tourism",
                "Event Management",
                "Airline Management",
                "Hospital Administration",
                "Culinary Arts"
            ]
        }
    }

    

    # ------------------ MAIN INTERACTIVE PANEL ------------------
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

        # Aggregate Percentage input textbox
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
                label="⚡ Career Readiness Score",
                value=f"{predicted_score:.1f} / 100"
            )
            st.progress(int(predicted_score))

        with res_col2:

            recommendation = course_recommendations.get(
                predicted_course,
                {
                    "best": predicted_course,
                    "others": []
                }
            )

            st.metric(
                label="🎯 Best Recommended Course",
                value=recommendation["best"]
            )

        st.markdown("---")

        st.subheader("📚 Other Suitable Course Options")

        if recommendation["others"]:
            for course in recommendation["others"]:
                st.write(f"✅ {course}")
        else:
            st.info("No additional course recommendations available.")

        st.markdown("---")

        st.subheader("💡 Career Guidance")

        st.info(
            f"""
        Based on your academic profile, skills, and interests, **{recommendation['best']}**
        is the most suitable course for you.

        You may also consider the alternative courses listed above, as they are closely
        related to your strengths and career goals.
        """      )
