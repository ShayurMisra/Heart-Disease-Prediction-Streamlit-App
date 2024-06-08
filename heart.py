import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Question 1

df = pd.read_csv('heart.csv')

split_columns = df['age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'].str.split(';', expand=True)
df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']] = split_columns

df.drop(columns=['age;sex;cp;trestbps;chol;fbs;restecg;thalach;exang;oldpeak;slope;ca;thal;target'], inplace=True)

with sqlite3.connect('heart_database.db') as conn:
    
    df.to_sql('heart_table', conn, if_exists='replace', index=False)

#Question 2.1a

with sqlite3.connect('heart_database.db') as conn:
    query = "SELECT * FROM heart_table"
    df = pd.read_sql(query, conn)

missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

numeric_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

df.fillna(df.mean(), inplace=True)

#Question 2.1b

categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()
for i, column in enumerate(categorical_columns):
    sns.countplot(x=column, hue='target', data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {column} by Target')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')
plt.tight_layout()
plt.show()

#Question 2.1c

numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()
for i, column in enumerate(numeric_columns):
    sns.boxplot(x='target', y=column, data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {column} by Target')
    axes[i].set_xlabel('Target')
    axes[i].set_ylabel(column)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Question 3.1

X = df.drop(columns=['target'])
y = df['target']

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, 'preprocessor.pkl')

# Question 3.2

random_forest = RandomForestClassifier(random_state=42)
svm = SVC(kernel='rbf', random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

random_forest.fit(X_train_processed, y_train)
svm.fit(X_train_processed, y_train)
gradient_boosting.fit(X_train_processed, y_train)

y_pred_rf = random_forest.predict(X_test_processed)
y_pred_svm = svm.predict(X_test_processed)
y_pred_gb = gradient_boosting.predict(X_test_processed)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"Gradient Boosting Accuracy: {accuracy_gb:.4f}")

if accuracy_gb > accuracy_rf and accuracy_gb > accuracy_svm:
    best_model = gradient_boosting
    best_model_name = "Gradient_Boosting_Model.pkl"
elif accuracy_rf > accuracy_svm:
    best_model = random_forest
    best_model_name = "Random_Forest_Model.pkl"
else:
    best_model = svm
    best_model_name = "SVM_Model.pkl"

joblib.dump(best_model, best_model_name)
print(f"Saved the best model ({best_model_name}) to disk.")