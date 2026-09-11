import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Activity AI",
    page_icon="🏃",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #777;
}

.prediction-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    background-color: #f5f7fa;
    margin-top: 20px;
    margin-bottom: 20px;
}

.prediction-text {
    font-size: 32px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load("har_svm_model.joblib")


# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("sample_data.csv")


model = load_model()
X_test = load_data()


# ACTIVITY MAPPING

activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


# SIDEBAR

st.sidebar.title("🏃 Activity AI")

st.sidebar.markdown(
    "Human Activity Classification System"
)

st.sidebar.divider()

st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Model Performance",
        "📋 Raw Data"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Dataset: UCI Human Activity Recognition Using Smartphones"
)


# DASHBOARD

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="title">🏃 Activity AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Human Activity Classification using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.header("📌 System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Test Accuracy",
            "96.17%"
        )

    with col2:
        st.metric(
            "Features",
            "561"
        )

    with col3:
        st.metric(
            "Algorithm",
            "SVM"
        )

    with col4:
        st.metric(
            "Kernel",
            "Linear"
        )

    st.divider()

    st.header("🧠 Recognized Activities")

    activities = pd.DataFrame({
        "Activity": [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
        ],
        "Description": [
            "Person walking normally",
            "Person walking upstairs",
            "Person walking downstairs",
            "Person sitting",
            "Person standing",
            "Person lying down"
        ]
    })

    st.dataframe(
        activities,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.header("⚙️ How the System Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📱 Sensor Data")
        st.write(
            "Accelerometer and gyroscope measurements "
            "are collected from smartphone sensors."
        )

    with col2:
        st.markdown("### 🔢 561 Features")
        st.write(
            "Time-domain and frequency-domain features "
            "are extracted from sensor signals."
        )

    with col3:
        st.markdown("### ⚖️ Scaling")
        st.write(
            "StandardScaler normalizes the feature values "
            "before classification."
        )

    with col4:
        st.markdown("### 🤖 SVM")
        st.write(
            "A Linear Support Vector Machine predicts "
            "the human activity."
        )

    st.divider()

    st.header("📈 Model Configuration")

    config = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Kernel",
            "C",
            "Cross Validation",
            "Scoring"
        ],
        "Value": [
            "Support Vector Machine",
            "Linear",
            "0.1",
            "5-Fold",
            "Accuracy"
        ]
    })

    st.dataframe(
        config,
        width="stretch",
        hide_index=True
    )


# PREDICTION

elif page == "🔮 Prediction":

    st.title("🔮 Activity Prediction")

    st.write(
        "Use the trained Linear SVM model to predict "
        "the human activity from sensor-derived features."
    )

    st.divider()

    st.subheader("📄 Sample CSV Data")

    st.write(
        f"Dataset contains {X_test.shape[0]} sample(s) "
        f"and {X_test.shape[1]} features."
    )

    with st.expander("🔍 View Sample CSV Data"):

        st.dataframe(
            X_test,
            width="stretch",
            height=400
        )

    st.divider()

    if len(X_test) > 1:

        sample_index = st.slider(
            "Select Test Sample",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0
        )

    else:

        sample_index = 0

        st.info(
            "One test sample is currently available."
        )

    sample = X_test.iloc[[sample_index]]

    st.divider()

    if st.button(
        "🚀 Predict Activity",
        width="stretch"
    ):

        prediction = model.predict(
            sample.to_numpy()
        )[0]

        predicted_activity = activity_map[
            int(prediction)
        ]

        st.markdown(
            f"""
            <div class="prediction-card">
                <div>Predicted Activity</div>
                <div class="prediction-text">
                    {predicted_activity}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction generated successfully."
        )

        st.info(
            "The prediction was generated using the "
            "trained Linear SVM model."
        )


# MODEL PERFORMANCE

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Performance of the trained Linear SVM model "
        "on the UCI HAR test dataset."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Test Accuracy",
            "96.17%"
        )

    with col2:
        st.metric(
            "CV Accuracy",
            "93.58%"
        )

    with col3:
        st.metric(
            "C",
            "0.1"
        )

    with col4:
        st.metric(
            "Kernel",
            "Linear"
        )

    st.divider()

    st.header("🎯 Classification Performance")

    performance_df = pd.DataFrame({
        "Activity": [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
        ],
        "Precision": [
            0.96,
            0.96,
            0.99,
            0.96,
            0.91,
            1.00
        ],
        "Recall": [
            1.00,
            0.96,
            0.95,
            0.89,
            0.97,
            1.00
        ],
        "F1 Score": [
            0.98,
            0.96,
            0.97,
            0.92,
            0.94,
            1.00
        ]
    })

    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.header("📈 Classification Metrics")

    metric_df = performance_df.melt(
        id_vars="Activity",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        metric_df,
        x="Activity",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1.05],
        title="Precision, Recall and F1 Score"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.divider()

    st.header("🔥 Confusion Matrix")

    confusion = np.array([
        [495, 0, 1, 0, 0, 0],
        [16, 453, 2, 0, 0, 0],
        [6, 15, 399, 0, 0, 0],
        [0, 2, 0, 436, 53, 0],
        [0, 0, 0, 18, 514, 0],
        [0, 0, 0, 0, 0, 537]
    ])

    labels = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    fig = px.imshow(
        confusion,
        x=labels,
        y=labels,
        text_auto=True,
        aspect="auto",
        labels={
            "x": "Predicted Activity",
            "y": "Actual Activity",
            "color": "Count"
        },
        title="Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.info(
        "The main classification confusion occurs between "
        "SITTING and STANDING."
    )


# RAW DATA

elif page == "📋 Raw Data":

    st.title("📋 Raw Test Data")

    st.write(
        "The model receives 561 sensor-derived features "
        "as input."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            X_test.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            X_test.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(X_test.isnull().sum().sum())
        )

    st.divider()

    st.subheader("📊 Dataset")

    st.dataframe(
        X_test,
        width="stretch",
        height=500
    )

    st.divider()

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        X_test.describe(),
        width="stretch"
    )


# FOOTER

st.divider()

st.caption(
    "Activity AI | UCI HAR Dataset | Linear SVM | "
    "Test Accuracy: 96.17%"
)