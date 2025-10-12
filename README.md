# 🎓 Student Performance Prediction System

A machine learning system that predicts whether a student will pass or fail based on demographic, social, and academic factors. Built with Python, scikit-learn, and Streamlit.

**Live Demo:** [https://student-success-ml-system.streamlit.app](https://student-success-ml-system.streamlit.app)

---

## 📊 Project Overview

This system analyzes 30+ student factors to predict academic outcomes, helping educators identify at-risk students and intervene early. The model was trained on 395 students from the UCI Student Performance dataset with **78% F1-score accuracy**.

### Key Features
- **Real-time predictions** via interactive web interface
- **Interpretable results** showing key performance factors
- **Feature importance analysis** explaining model decisions
- **Actionable insights** for educators and parents

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| **Best Model** | Tuned Decision Tree |
| **F1-Score** | 78.05% |
| **Accuracy** | 65.82% |
| **Recall** | 90.57% |
| **Test Set Size** | 79 students |

### Top Predictive Features
1. **Previous Failures** (16%) - Strongest predictor of outcomes
2. **Absences** (11%) - Class attendance matters significantly
3. **Going Out Frequency** (7%) - Social engagement affects performance
4. **Age** (5%) - Age-related maturity factor
5. **Mother's Job** (4%) - Family background influence

---

## 🛠️ Technology Stack

**Machine Learning:**
- Python 3.13
- scikit-learn 1.4.2 (Random Forest, Decision Tree, GridSearchCV)
- pandas 2.2.2 (data processing)
- numpy 1.26.4 (numerical computing)

**Data Visualization:**
- matplotlib 3.8.4
- seaborn 0.13.2

**Web Interface:**
- Streamlit 1.31.0 (interactive dashboard)

**Development & Deployment:**
- Git & GitHub (version control)
- Streamlit Cloud (free hosting)

---

## 📁 Project Structure

```
student-success-ml-system/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
│
├── data/
│   └── student-mat.csv            # UCI Student Performance dataset
│
├── src/
│   ├── train.py                   # Model training script
│   └── predict.py                 # Prediction module
│
├── models/
│   ├── random_forest_model.pkl    # Trained Random Forest
│   ├── decision_tree_model.pkl    # Trained Decision Tree
│   └── label_encoders.pkl         # Categorical encoders
│
├── notebooks/
│   └── analysis.ipynb             # Full analysis & EDA
│
└── images/
    └── [visualizations]           # Project screenshots
```

---

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/student-success-ml-system.git
cd student-success-ml-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models (generates .pkl files)
python src/train.py

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Online Demo

Visit the live app: [https://student-success-ml-system.streamlit.app](https://student-success-ml-system.streamlit.app)

No installation required - just fill in student details and click "Predict".

---

## 📊 How It Works

### Data Input
Users enter 30 student attributes across three categories:
- **Academic:** School type, study time, past failures, absences
- **Family:** Family size, parental education, parent status
- **Social:** Relationships, alcohol consumption, activities, health

### Model Pipeline
1. **Data Preprocessing:** Categorical encoding with LabelEncoder
2. **Feature Engineering:** 30 features normalized and prepared
3. **Model Selection:** Random Forest + Decision Tree comparison
4. **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
5. **Prediction:** Binary classification (Pass/Fail at grade ≥10)

### Output
- **Pass/Fail Prediction** with confidence percentage
- **Top Contributing Factors** (feature importance)
- **Performance Indicators** (risk factors and positive signs)
- **Actionable Recommendations** for educators

---

## 🔍 Key Insights from Analysis

**Finding 1:** Previous failures are the #1 predictor
- Students with 0 failures: **75% pass rate**
- Students with failures: **37% pass rate**

**Finding 2:** Attendance is critical
- Each additional absence impacts outcome significantly
- Chronic absenteeism (<5 absences) strongly correlates with success

**Finding 3:** Family support matters
- Parental education levels influence student outcomes
- Family educational support improves performance

**Finding 4:** Balance is important
- Extreme study time (<2 hours or >10 hours) shows different patterns
- Social engagement (2-5 hours/week going out) shows optimal outcomes

---

## 📈 Model Performance Details

### Training Process
- **Dataset:** 395 Portuguese mathematics students
- **Train/Test Split:** 80/20 stratified
- **Cross-Validation:** 5-fold GridSearchCV
- **Metric:** F1-score (balances precision & recall)

### Decision Tree (Best Model)
- Max depth: 3
- Split criterion: Entropy
- Hyperparameters tuned via GridSearchCV
- **Recall: 90.57%** (catches at-risk students)

### Random Forest
- 100 trees, max depth: 10
- Good precision but slightly lower recall
- Better for general predictions

### Why Not Perfect Accuracy?
Student performance depends on many factors beyond these 30 variables (teacher quality, personal motivation, external events, etc.). 78% F1-score represents realistic predictive power.

---

##  Use Cases

**For Educators:**
- Identify at-risk students early in the semester
- Allocate tutoring resources effectively
- Monitor intervention effectiveness

**For Parents:**
- Understand factors affecting their child's performance
- Identify areas for improvement
- Track progress over time

**For Researchers:**
- Analyze student success factors
- Compare educational strategies
- Generate hypotheses for further study

---

## 📚 Dataset Source

**UCI Student Performance Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/student+performance)
- Citation: P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." In Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC) 2008. pp. 5-12.
- Students: 395 (Mathematics course)
- Features: 30 demographic, social, and school-related attributes
- Target: Final grade (converted to Pass/Fail at grade ≥10)

---

## 🔄 Model Deployment & Updates

The model is deployed on **Streamlit Cloud** and automatically redeploys when the GitHub repository is updated.

To update the model:
1. Retrain locally: `python src/train.py`
2. Commit changes: `git add . && git commit -m "Update trained models"`
3. Push to GitHub: `git push origin main`
4. Streamlit Cloud automatically redeploys (2-3 minutes)

---

## 📝 Files Overview

**app.py**
- Streamlit interface
- User input form with 30 fields
- Real-time predictions
- Feature importance visualization

**src/train.py**
- Data loading and preprocessing
- Model training with GridSearchCV
- Model persistence (pickle files)
- Evaluation metrics calculation

**src/predict.py**
- Model loading utilities
- Prediction function
- Feature importance extraction

**requirements.txt**
- All Python dependencies with versions

---

## 🤝 Contributing

Suggestions for improvement:
- Add more student attributes
- Implement SHAP values for better interpretability
- Develop multi-class prediction (grade tiers instead of binary)
- Create teacher dashboard for batch predictions
- Add model explanation with LIME

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Attribution:** Dataset from UCI Machine Learning Repository (Cortez & Silva, 2008)

---

## 👨‍💻 Author

**SMJ**

- GitHub: [@yourusername](https://github.com/Ph0enix22)
- LinkedIn: [your-profile](https://linkedin.com/in/syeda-midhath)
- Email: syedamidhath159@gmail.com

---

## 📞 Support

For questions or issues:
1. Check the [GitHub Issues](https://github.com/Ph0enix22/student-success-ml-system/issues)
2. Review the [Streamlit documentation](https://docs.streamlit.io/)
3. Consult the [scikit-learn documentation](https://scikit-learn.org/)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Streamlit for the amazing web framework
- scikit-learn for excellent ML tools
- Python community for all open-source libraries

---
**Made with ❤️ using Python, Machine Learning, and Open Source**

Last updated: October 2025
