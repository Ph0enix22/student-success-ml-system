"""
Student Performance Prediction - Prediction Module
Functions for loading models and making predictions.
"""

import pickle
import pandas as pd
import os

def load_models():
    """Load saved models and encoders."""
    if not os.path.exists('models/random_forest_model.pkl'):
        raise FileNotFoundError(
            "Models not found! Please run 'python src/train.py' first."
        )
    
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

def predict_performance(student_data, use_random_forest=True):
    """
    Predict if a student will pass or fail.
    
    Parameters:
    -----------
    student_data : dict
        Dictionary containing student features
    use_random_forest : bool
        If True, use Random Forest model, else use Decision Tree
    
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
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                print(f"Warning: Unknown category in {col}: {df[col].values[0]}")
                # Use the most common class
                df[col] = 0
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return prediction, probability

def get_feature_importance(top_n=10):
    """Get top N most important features from the Random Forest model."""
    model, _ = load_models()
    
    # Get feature names (need to load data to get column names)
    import pandas as pd
    df = pd.read_csv('data/student-mat.csv', sep=';')
    df['pass'] = (df['G3'] >= 10).astype(int)
    df_processed = df.drop(columns=['G1', 'G2', 'G3'])
    
    # Encode categoricals to get feature names
    from sklearn.preprocessing import LabelEncoder
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
    
    X = df_processed.drop('pass', axis=1)
    feature_names = X.columns
    
    # Get importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df.head(top_n)

# Example usage and testing
if __name__ == "__main__":
    print("Testing prediction module...")
    
    # Sample student data
    sample_student = {
        'school': 'GP',
        'sex': 'F',
        'age': 17,
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'T',
        'Medu': 3,
        'Fedu': 3,
        'Mjob': 'teacher',
        'Fjob': 'other',
        'reason': 'course',
        'guardian': 'mother',
        'traveltime': 1,
        'studytime': 2,
        'failures': 0,
        'schoolsup': 'yes',
        'famsup': 'no',
        'paid': 'no',
        'activities': 'no',
        'nursery': 'yes',
        'higher': 'yes',
        'internet': 'yes',
        'romantic': 'no',
        'famrel': 4,
        'freetime': 3,
        'goout': 3,
        'Dalc': 1,
        'Walc': 2,
        'health': 4,
        'absences': 4
    }
    
    try:
        pred, prob = predict_performance(sample_student)
        print(f"\nPrediction: {'PASS' if pred == 1 else 'FAIL'}")
        print(f"Confidence: {prob:.2%}")
        
        print("\nTop 10 Important Features:")
        print(get_feature_importance())
        
    except FileNotFoundError as e:
        print(f"\n⚠️ Error: {e}")
        print("Please run 'python src/train.py' first to generate the models.")
