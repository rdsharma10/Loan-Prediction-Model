Loan Prediction Model Using Decision Tree and Naïve Bayes Classifier
Introduction
The Loan Prediction Model aims to predict whether a loan applicant will be approved or not based on various financial and personal attributes. The dataset includes details such as Applicant Income, Loan Amount, Credit History, Marital Status, and Employment Type. The model follows a machine learning pipeline consisting of data preprocessing, exploratory data analysis (EDA), handling missing values, feature engineering, training machine learning models, and evaluating their accuracy.

Data Preprocessing & Exploration
Loading the Dataset:
The dataset is read into a pandas DataFrame, and its first few rows are displayed using dataset.head(). Basic dataset statistics such as shape, summary statistics (dataset.describe()), and missing values (dataset.isnull().sum()) are analyzed.

Exploratory Data Analysis (EDA):

Categorical Data Analysis: pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins=True) is used to analyze the impact of credit history on loan approval.
Numerical Data Analysis:
Boxplots (dataset.boxplot(column='ApplicantIncome') and dataset.boxplot(column='LoanAmount')) help in identifying outliers.
Histograms (dataset['ApplicantIncome'].hist(bins=20)) visualize the distribution of numeric variables.
Handling Missing Values:

Mode imputation is applied for categorical variables such as Gender, Married, Dependents, and Credit_History.
The mean value is used for imputing missing values in LoanAmount.
Log transformations (np.log(dataset['TotalIncome'])) are applied to TotalIncome to normalize its distribution.
Feature Engineering & Data Preparation
Creating New Features:

A new feature, TotalIncome, is derived by adding ApplicantIncome and CoapplicantIncome.
A log-transformed version, TotalIncome_log, is created to reduce skewness.
Feature Selection:

The feature matrix X is created using a subset of relevant columns: x = dataset.iloc[:,np.r_[1:5,9:11,13:15]].values.
The target variable y is extracted from the Loan Status column.
Train-Test Split:

The dataset is split into training and testing sets using train_test_split() with an 80-20 split.
Encoding Categorical Variables:

LabelEncoder is applied to categorical columns to convert them into numerical values.
Feature Scaling:

StandardScaler is used to normalize the feature values for better model performance.
Model Training & Prediction
Decision Tree Classifier:
A Decision Tree Classifier is trained on the dataset using entropy as the splitting criterion:

DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(x_train, y_train)
Predictions are made on the test data, and accuracy is calculated using metrics.accuracy_score(y_pred, y_test).

Naïve Bayes Classifier:
A Gaussian Naïve Bayes model is trained as an alternative to the Decision Tree:


NBClassifier = GaussianNB()
NBClassifier.fit(x_train, y_train)
Predictions are made on the test dataset, and accuracy is evaluated.

Testing on New Data
The model is then tested on an unseen dataset (test.csv):

Missing values are handled similarly to the training dataset.
Feature transformations (log transformation and feature selection) are applied.
Label encoding and standardization are performed.
Predictions are made using the trained Naïve Bayes classifier.
Conclusion
The Loan Prediction Model is a binary classification problem solved using Decision Tree and Naïve Bayes algorithms. The model preprocesses data, handles missing values, performs feature engineering, and applies machine learning classifiers to make predictions. This approach is useful for banks and financial institutions to automate loan approval decisions efficiently.
