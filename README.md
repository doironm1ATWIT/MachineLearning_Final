# MachineLearning_Final


## Dataset

The dataset used in this project is based on heart disease fro Kaggle (https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog). There are 270 entries in the dataset. There are 13 attributes in this data set with one target variable. The target variable is labeled with either '0' no heart disease or '1' heart disease is detected. 

The 13 attributes are the following:
Age - Patient age in years
Sex - Gender of the Patient (Male: 1, Female: 0)
cp - type of chest pain experienced (0 : typical angina, 1 : atypical angina, 2 : non - anginal pain, 3: asymptomatic)
trestbps - patient's level of blood pressure at rest. 
chol - serum cholestrol
fbs - blood sugar lebels on fasting.
restecg - result of electrocardiogram while at rest (0: Normal, 1: having ST-T wave abnormality)
thalach- Max heart rate achieved.
exang - Angina induced by exercise (0 : no, 1: yes)
oldpeak - exercise induced ST-depression in relative with state of rest
slope - ST segment measured in terms of slope during peak exercise.
ca - number of major blood vessels
thal - a blood disorder valled thalassmia

Questions to answer about dataset:

Is the dataset balanced?
The target variable labeled with 1 has 120 entries.
The target variable labeled with 0 has 150 entries.

I would consider this dataset to be relatively balanced. There is not an overwhelming amount of either 0 or 1. 

Any correlation between features?

![data_screenshot](target_variable_vs_thalach.png)

There looks to be a small correlation between the target variable and thalach. It looks like more people who have a lower thalach level are diagnosed with heart disease. 

![data_screenshot](target_vs_trestbps_scatter.png)
It looks like there is a small correlation between the target variable and trestbps. The higher the trestbps, the more likely there is heart disease detected.


Are there any outliers?

insert trestbps boxplot image


Looking at this boxplot, there seems to be quite a few outliers for the attribute trestbps. This could cause an issue in our machine learning model predictions, but we will leave them in the dataset for now. 


## Models

The three models used in this analysis are Logistic Regression, Support Vector Classifier, and Random Forests. Below are the results without hyperparameter tuning.

| Model              | Test Accuracy Score | Train Accuracy Score | ROC AUC Score |
| ------------------ | ------------------- | -------------------- | ------------- |
| Logistic Regression| 0.86                | 0.84                 | 0.85          |
| SVC                | 0.70                | 0.63                 | 0.67          |
| Random Forest      | 0.85                | 1.0                  | 0.84          |




## Baseline Performance
