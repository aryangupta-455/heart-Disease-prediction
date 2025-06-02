# heart-Disease-prediction

The dataset consists of 303 observations and the following features:

age: Age of the person
sex: Gender (1 = male, 0 = female)
cp: Chest pain type (0-3)
trtbps: Resting blood pressure (mm Hg)
chol: Serum cholesterol (mg/dl)
fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
restecg: Resting electrocardiographic results (0-2)
thalachh: Maximum heart rate achieved
exng: Exercise-induced angina (1 = yes, 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slp: Slope of the peak exercise ST segment (0-2)
caa: Number of major vessels colored by fluoroscopy (0-3)
thall: Thalassemia (1-3)
output: Target variable (1 = heart disease present, 0 = no heart disease)

# Project-features
Model Building: Logistic Regression is used as the prediction model.
Web Application: A streamlit-based interface for user input and prediction.


# File Structure
app.py: The main Python script for running the strreamlit application.
heart_disease_prediction.csv: The dataset file sourced from Kaggle.
requirements.txt: List of required Python libraries.
# Future Work
Implementing additional machine learning models.
Enhancing the UI for better user interaction.
Using a larger dataset for improved generalization.
