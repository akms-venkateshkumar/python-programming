# Data Analytics — Curriculum & Sample Exercises

This file contains a progressive set of short lessons and code samples for students, starting from fundamentals and moving to advanced topics in data analytics. Each lesson includes: objective, key concepts, a short explanation, a Python code sample, and an exercise.

Prerequisites
- Basic Python (variables, lists, functions)
- Recommended packages: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels
- Install: pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter

Datasets to use
- Iris (sklearn.datasets)
- Titanic (Kaggle or seaborn/other CSV)
- Example CSVs (students can create their own or use UCI datasets)

1) Introduction to Data Analytics
Objective: Understand the data analytics workflow and common terminology.
Key concepts: data collection, cleaning, exploration, modeling, interpretation, communication.

Exercise: Describe a simple analytics workflow to answer "Which product sold best last quarter?" Outline data sources and steps.

2) Working with CSVs and pandas basics
Objective: Load CSVs, inspect data, basic selection and filtering.

Sample:
```python
import pandas as pd
# load
df = pd.read_csv('data/titanic.csv')
# inspect
print(df.shape)
print(df.head())
# select
survivors = df[df['Survived']==1]
print(survivors[['Name','Sex','Age']].head())
```
Exercise: Load a CSV, show the number of missing values per column, and print first 5 rows of passengers older than 50.

3) Data types, conversion, and missing values
Objective: Recognize dtypes, convert types, handle NaNs.

Sample:
```python
# dtypes
print(df.dtypes)
# convert Age to numeric if needed
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# missing values
print(df.isna().sum())
# simple imputation
df['Age'].fillna(df['Age'].median(), inplace=True)
```
Exercise: Impute missing 'Embarked' values with the mode and create a boolean column 'AgeMissing'.

4) Exploratory Data Analysis (EDA)
Objective: Summarize distributions and relationships.
Key tools: describe(), value_counts(), groupby(), pivot_table, visualizations.

Sample:
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()
# relationship
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.show()
```
Exercise: Show survival rate by Sex and Pclass using groupby and a bar plot.

5) Data visualization principles
Objective: Learn good charts, labeling, and when to use which plot.

Sample:
```python
# bar chart of counts
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Class')
plt.show()
```
Exercise: Create a faceted plot of Age distribution by Sex.

6) Feature engineering
Objective: Create useful features (e.g., extract titles from names, bucket ages).

Sample:
```python
# extract Title
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
# age bins
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,20,40,60,120], labels=['Child','Teen','Adult','Mid','Senior'])
```
Exercise: Create a family size feature from SibSp + Parch, then examine survival rate by family size.

7) Basic statistics and probability
Objective: Mean, median, variance, standard error, confidence intervals, distributions.

Sample:
```python
import numpy as np
ages = df['Age'].dropna()
print('mean', ages.mean())
print('median', ages.median())
print('std', ages.std())
# bootstrap a 95% CI for mean
means = [ages.sample(frac=1, replace=True).mean() for _ in range(1000)]
print(np.percentile(means, [2.5,97.5]))
```
Exercise: Compute a 95% CI for the median Fare using bootstrap.

8) Hypothesis testing
Objective: t-tests, chi-squared tests, when to use which test.

Sample:
```python
from scipy import stats
# compare age between survivors and non-survivors
a = df[df['Survived']==1]['Age'].dropna()
b = df[df['Survived']==0]['Age'].dropna()
print(stats.ttest_ind(a, b, nan_policy='omit'))
```
Exercise: Test whether survival is independent of Embarked (chi-squared test).

9) Correlation and covariance
Objective: Pearson, Spearman, heatmaps, multicollinearity.

Sample:
```python
corr = df[['Age','Fare','Pclass','Survived']].corr()
sns.heatmap(corr, annot=True)
plt.show()
```
Exercise: Compute Spearman correlation between Fare and Age; interpret result.

10) Introduction to machine learning: supervised vs unsupervised
Objective: Understand modeling goals: prediction, classification, clustering.

11) Regression (linear regression)
Objective: Fit, interpret coefficients, evaluate.

Sample:
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df[['Age']].fillna(df['Age'].median())
y = df['Fare']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print('coef', model.coef_, 'intercept', model.intercept_)
print('R^2', model.score(X_test, y_test))
```
Exercise: Fit a regression of Fare on Pclass and AgeGroup (one-hot encode AgeGroup). Interpret coefficients.

12) Classification (logistic regression, decision trees)
Objective: Predict binary outcomes and evaluate with accuracy, precision, recall, ROC-AUC.

Sample:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
features = ['Pclass','Age']
X = df[features].fillna(df['Age'].median())
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
clf = LogisticRegression(max_iter=200).fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))
probs = clf.predict_proba(X_test)[:,1]
print('AUC', roc_auc_score(y_test, probs))
```
Exercise: Train a RandomForestClassifier and compare ROC-AUC with logistic regression.

13) Model validation and cross-validation
Objective: Use k-fold CV, avoid overfitting, grid search hyperparameters.

Sample:
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(rf, X.fillna(0), y, cv=5, scoring='roc_auc')
print(scores, scores.mean())
```
Exercise: Use GridSearchCV to tune max_depth and n_estimators for RandomForest.

14) Feature selection and regularization
Objective: L1/L2 regularization, recursive feature elimination, feature importance.

Sample:
```python
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=1000).fit(X.fillna(0), y)
print('coefs', clf.coef_)
```
Exercise: Use SelectFromModel with L1-penalized logistic regression to pick features.

15) Dimensionality reduction (PCA, t-SNE)
Objective: Reduce dimensionality, visualize high-dim data.

Sample:
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)
plt.scatter(X2[:,0], X2[:,1], c=iris.target)
plt.title('Iris PCA')
plt.show()
```
Exercise: Use t-SNE to visualize MNIST subset (or iris) and compare with PCA.

16) Clustering (k-means, hierarchical)
Objective: Unsupervised grouping and cluster validation.

Sample:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_[:10])
```
Exercise: Compute silhouette score for different k and pick the best k.

17) Time series basics
Objective: Date parsing, resampling, rolling windows, decomposition.

Sample:
```python
# assume df_ts has columns Date, Value
# df_ts['Date'] = pd.to_datetime(df_ts['Date'])
# df_ts.set_index('Date', inplace=True)
# resample monthly
# monthly = df_ts['Value'].resample('M').mean()
# monthly.plot()
```
Exercise: Decompose a time series into trend/seasonality using statsmodels seasonal_decompose.

18) Advanced topics: pipelines, production, big data
Objective: Build pipelines, prepare models for production, basics of Spark, and deployment.

Sample: scikit-learn pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                 ('scale', StandardScaler()),
                 ('clf', RandomForestClassifier(n_estimators=100))])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```
Exercise: Save a trained pipeline with joblib and write a small Flask app to serve predictions.

19) Model monitoring, ethics, reproducibility
Objective: Bias detection, fairness metrics, experiment tracking (MLflow), seeds and deterministic runs.

Exercise: Set random seeds for numpy and sklearn, log runs with MLflow, and plot model drift metrics over time.

20) Projects & capstone ideas
- End-to-end Titanic reproduction and report
- Sales forecasting for a store (time series)
- Customer segmentation with clustering
- Sentiment analysis on tweets (NLP)

How to use these samples
- Turn each lesson into a Jupyter notebook with the sample code and exercise cells.
- Provide small datasets in data/ or download scripts.
- Encourage students to write short reports interpreting results and visualizations.

Next steps I can take for this repo
- Split this into per-lesson Jupyter notebooks under features/copilot/plans/notebooks/
- Add sample datasets (small CSVs) and data download scripts
- Add automated tests or solution notebooks

