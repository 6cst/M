**EXPERIMENTx3 - MNIST DATASET BY 229X1A2856**



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score




# 1. Import MNIST dataset
mnist_digits = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist_digits["data"], mnist_digits["target"].astype(int)


# 2. Data Preprocessing
print("Dataset Info:")
print(pd.DataFrame(x).info())
print("\nHead of Dataset:")
print(pd.DataFrame(x).head())
print("\nTail of Dataset:")
print(pd.DataFrame(x).tail())
print("\nChecking for Missing Values:")
print(pd.DataFrame(x).isnull().sum().sum())
print("\nChecking for Duplicates:")
print(pd.DataFrame(x).duplicated().sum())
print("\nStatistical Summary:")
print(pd.DataFrame(x).describe())

# 3. Shape of Data
print("\nShape of Data:", x.shape)
print("Shape of Target:", y.shape)

# 4. Sample Digits Visualization
sample_a = x[200].reshape(28, 28)
sample_b = x[25].reshape(28, 28)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_a, cmap='gray')
plt.title("Sample Digit 200")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sample_b, cmap='gray')
plt.title("Sample Digit 25")
plt.axis("off")
plt.show()

# 5. Statistical Methods
print("\nMean:", np.mean(x))
print("\nMedian:", np.median(x))
print("\nStandard Deviation:", np.std(x))
print("\nMinimum:", np.min(x))
print("\nMaximum:", np.max(x))


# 6. Data Visualization - Bar Graph & Area Chart
y = y.astype(int)  # Ensure y is integer type to avoid TypeError

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(10), np.bincount(y, minlength=10), color='skyblue')
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("Digit Distribution - Bar Chart")

plt.subplot(1, 2, 2)
plt.fill_between(range(10), np.bincount(y, minlength=10), color='lightcoral', alpha=0.5)
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("Digit Distribution - Area Chart")
plt.show()

# 7. Splitting Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# 8. Train SGD Classifier
model = SGDClassifier(random_state=42)
model.fit(x_train, y_train)

# 9. Predictions
y_pred = model.predict(x_test)

# 10. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 11. Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# 12. Cross-Validation
cv_scores = cross_val_score(model, x_train, y_train, cv=3, scoring="accuracy")
print("Cross-Validation Scores:", cv_scores)


EXPERIMENTx4 KNN CLASSIFIER BY 229X1A2856

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
# Removing leading and trailing spaces in column names
df.columns = df.columns.str.strip()


# Removing null values
df.dropna(inplace=True)


print(df.info())  # 3.1
print(df.head())  # 3.2
print(df.tail())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())

# 4. Statistical Methods
print("Mean:", df.mean())
print("Median:", df.median())
print("Standard Deviation:", df.std())
print("Variance:", df.var())
print("Correlation:", df.corr())

# 5. Data Visualization
# Area plot
plt.figure(figsize=(10, 6))
df[['GRE_Score', 'TOEFL_Score', 'CGPA']].plot(kind='area', alpha=0.4, colormap='coolwarm')
plt.xlabel("Index")
plt.ylabel("Scores")
plt.title("Area Plot of Key Admission Factors")
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
df['CGPA'].hist(bins=20, color='skyblue', edgecolor='black')
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.title("Distribution of CGPA")
plt.show()

# 6. Data Splitting
X = df.drop(columns=['Chance_of_Admit'])
y = (df['Chance_of_Admit'] >= 0.7).astype(int)  # Convert to binary classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 7. Standardizing Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Model Training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 9. Predictions and Confusion Matrix
y_pred = knn.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Accuracy, Precision, Recall, F1-Score
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 11. Cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

EXPx5 BY 229X1A2856

# Step 1: Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 2: Load the dataset
df = pd.read_csv("/content/Admission_prediction.csv")

# Step 3: Data Preprocessing
# 3.1 Display dataset information
print(df.info())

# 3.2 Display first and last rows
print(df.head())
print(df.tail())

# Check for null values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())
# Removing null values
df.dropna(inplace=True)


# Describe dataset statistics
print(df.describe())

# Step 4: Statistical Methods
print("Mean:\n", df.mean())
print("Median:\n", df.median())
print("Mode:\n", df.mode().iloc[0])  # Mode can have multiple values, taking first
print("Standard Deviation:\n", df.std())
print("Variance:\n", df.var())

# Step 5: Data Visualization
# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram for one of the columns
plt.figure(figsize=(8,5))
sns.histplot(df["Chance_of_Admit"], bins=20, kde=True)
plt.title("Distribution of Chance_of_Admit")
plt.xlabel("Chance_of_Admit")
plt.ylabel("Frequency")
plt.show()


# Step 6: Data Splitting
X = df.drop(columns=["Chance_of_Admit"])  # Features
y = (df["Chance_of_Admit"] >= 0.5).astype(int)  # Convert into binary classification (0 or 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Naïve Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)


# Step 8: Predictions and Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

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
