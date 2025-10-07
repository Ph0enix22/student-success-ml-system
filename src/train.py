import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def load_and_preprocess_data(filepath):
    """Load and preprocess the student dataset."""
    df = pd.read_csv(filepath, sep=';')
    df['pass'] = (df['G3'] >= 10).astype(int)
    
    # Drop grade columns to avoid leakage
    df_processed = df.drop(columns=['G1', 'G2', 'G3'])
    
    # Encode categoricals
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
        label_encoders[column] = le
    
    X = df_processed.drop('pass', axis=1)
    y = df_processed['pass']
    
    return X, y, label_encoders

def train_models(X_train, y_train):
    """Train and tune models."""
    # Decision Tree
    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                      dt_params, cv=5, scoring='f1')
    dt.fit(X_train, y_train)
    
    # Random Forest
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42), 
                      rf_params, cv=5, scoring='f1')
    rf.fit(X_train, y_train)
    
    return dt.best_estimator_, rf.best_estimator_

if __name__ == "__main__":
    # Load data
    X, y, encoders = load_and_preprocess_data('data/student-mat.csv')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    print("Training models...")
    dt_model, rf_model = train_models(X_train, y_train)
    
    # Evaluate
    print("\nDecision Tree Results:")
    print(classification_report(y_test, dt_model.predict(X_test)))
    
    print("\nRandom Forest Results:")
    print(classification_report(y_test, rf_model.predict(X_test)))
    
    # Save models
    with open('models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print("\n✅ Models saved successfully!")
