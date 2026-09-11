# Human Activity Recognition

Machine learning project for recognizing human activities using smartphone sensor data.

#[Live Streamlit App] - (https://human-activity-recognition-pulc5pdwbcrbepri2zfm3f.streamlit.app/)

## Activities

The model predicts six activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Model

- Algorithm: Linear SVM
- Features: 561
- C: 0.1
- Cross-Validation: 5-Fold
- Test Accuracy: 96.17%

## Technologies

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib
- Plotly

## Files

- `app.py` - Streamlit application
- `HAR.ipynb` - Model training notebook
- `har_svm_model.joblib` - Trained model
- `sample_data.csv` - Sample test data
- `requirements.txt` - Required packages

## Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

UCI Human Activity Recognition Using Smartphones Dataset.
