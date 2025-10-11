"""
Student Performance Prediction - Model Training Script
This script trains Decision Tree and Random Forest models on student data.
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_and_preprocess_data(filepath):
    """Load and preprocess the student dataset."""
    print("Loading dataset...")
    df = pd.read_csv(filepath, sep=';')
    df['pass'] = (df['G3'] >= 10).astype(int)
    
    # Drop grade columns to avoid leakage
    df_processed = df.drop(columns=['G1', 'G2', 'G3'])
    
    # Encode categoricals
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    print(f"Encoding {len(categorical_columns)} categorical features...")
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
        label_encoders[column] = le
    
    X = df_processed.drop('pass', axis=1)
    y = df_processed['pass']
    
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, label_encoders

def train_models(X_train, y_train):
    """Train and tune models using GridSearchCV."""
    
    # Decision Tree
    print("\n🌳 Training Decision Tree with GridSearchCV...")
    dt_params = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42), 
        dt_params, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    dt.fit(X_train, y_train)
    print(f"✓ Best DT params: {dt.best_params_}")
    print(f"✓ Best DT CV F1-score: {dt.best_score_:.4f}")
    
    # Random Forest
    print("\n🌲 Training Random Forest with GridSearchCV...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        rf_params, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train, y_train)
    print(f"✓ Best RF params: {rf.best_params_}")
    print(f"✓ Best RF CV F1-score: {rf.best_score_:.4f}")
    
    return dt.best_estimator_, rf.best_estimator_

def save_models(dt_model, rf_model, encoders):
    """Save trained models and encoders."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("\n💾 Saving models...")
    with open('models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print("✓ Decision Tree saved")
    
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("✓ Random Forest saved")
    
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("✓ Label encoders saved")

def main():
    print("=" * 60)
    print("🎓 Student Performance Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    X, y, encoders = load_and_preprocess_data('data/student-mat.csv')
    
    # Split data
    print("\n📊 Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    dt_model, rf_model = train_models(X_train, y_train)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("📈 TEST SET EVALUATION")
    print("=" * 60)
    
    print("\n🌳 Decision Tree Results:")
    dt_pred = dt_model.predict(X_test)
    print(classification_report(y_test, dt_pred, target_names=['Fail', 'Pass']))
    print(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, dt_pred):.4f}")
    
    print("\n🌲 Random Forest Results:")
    rf_pred = rf_model.predict(X_test)
    print(classification_report(y_test, rf_pred, target_names=['Fail', 'Pass']))
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, rf_pred):.4f}")
    
    # Save models
    save_models(dt_model, rf_model, encoders)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to test the web interface")
    print("2. Models are saved in the 'models/' folder")
    print("3. Ready for deployment!")

if __name__ == "__main__":
    main()
