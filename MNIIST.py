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

#7 . Splitting Data
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







