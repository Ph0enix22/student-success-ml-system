import pickle
import pandas as pd
import numpy as np

def load_models():
    """Load saved models and encoders."""
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

def predict_performance(student_data):
    """
    Predict if a student will pass or fail.
    
    Parameters:
    -----------
    student_data : dict
        Dictionary containing student features
    
    Returns:
    --------
    prediction : int (0=Fail, 1=Pass)
    probability : float (probability of passing)
    """
    model, encoders = load_models()
    
    # Convert to DataFrame
    df = pd.DataFrame([student_data])
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
    sample_student = {
        'school': 'GP',
        'sex': 'F',
        'age': 17,
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 3,
        'Fedu': 3,
        'studytime': 2,
        'failures': 0,
        'absences': 4,
        # ... include all 30 features
    }
    
    pred, prob = predict_performance(sample_student)
    print(f"Prediction: {'Pass' if pred == 1 else 'Fail'}")
    print(f"Confidence: {prob:.2%}")
