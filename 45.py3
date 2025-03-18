import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Import dataset
df = pd.read_csv("/content/Admission_prediction.csv")

# 3. Data Preprocessing
print(df.info())  # 3.1

print(df.head())  # 3.2

print(df.tail())

print(df.isnull().sum())

# Removing null values
df.dropna(inplace=True)

print(df.isnull().sum())

print(df.describe())

print(df.duplicated().sum())

print(df.describe())

# 4. Statistical Methods
print("Mean:", df.mean())

print("Median:", df.median())

print("Standard Deviation:\n\n", df.std())
print("\nVariance:\n\n", df.var())
print("\nCorrelation:\n\n", df.corr())

print(df.columns) #To know column names..

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram for one of the columns
plt.figure(figsize=(8,5))
sns.histplot(df["Chance_of_Admit"], bins=20, kde=True)
plt.title("Distribution of Chance of Admit")
plt.xlabel("Chance of Admit")
plt.ylabel("Frequency")
plt.show()

# Step 6: Data Splitting
X = df.drop(columns=["Chance_of_Admit"])  # Features
y = (df["Chance_of_Admit"] >= 0.5).astype(int)  # Convert into binary classification (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Step 7: Train Naïve Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 8: Predictions and Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 9. Predictions and Confusion Matrix
y_pred =knn.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Step 10: Cross Validation (if possible)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

Here's the corrected code:

Labels and sizes for pie chart
labels = ["Admitted", "Not Admitted"]
sizes = [df[df["Chance_of_Admit"] > 0.7].shape[0], df[df["Chance_of_Admit"] < 0.7].shape[0]]

Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=148, colors=["skyblue", "lightcoral"])
plt.title("Admission Chances Distribution")
plt.show()

Data splitting
X = df.drop(columns=["Chance_of_Admit"])
y = df["Chance_of_Admit"].apply(lambda x: 1 if x > 0.7 else 0)

Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Standardizing data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Model training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

Predictions and confusion matrix
y_pred = knn.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

Accuracy, precision, recall, F1-score
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
