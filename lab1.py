# mushroom_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# 1. Завантаження даних
# ===============================
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = [
    'class','cap-shape','cap-surface','cap-color','bruises','odor',
    'gill-attachment','gill-spacing','gill-size','gill-color',
    'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',
    'stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color',
    'ring-number','ring-type','spore-print-color','population','habitat'
]
df = pd.read_csv(data_url, header=None, names=columns)

print("📊 Розмір датасету:", df.shape)
print("🧾 Колонки:", df.columns.tolist())

# ===============================
# 2. Опрацювання пропусків
# ===============================
df.replace("?", np.nan, inplace=True)
imputer = SimpleImputer(strategy="most_frequent")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# ===============================
# 3. Візуалізація
# ===============================
label_encoders = {}
df_encoded = df_imputed.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

corr = df_encoded.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Кореляційна матриця")
plt.tight_layout()
plt.savefig("heatmap.png")
print("✅ Heatmap збережено як heatmap.png")

for col in ['cap-shape', 'odor', 'gill-color']:
    plt.figure()
    df_imputed[col].value_counts().plot(kind="bar")
    plt.title(f"Розподіл {col}")
    plt.savefig(f"hist_{col}.png")

# Boxplot для прикладу
plt.figure(figsize=(8, 6))
sns.boxplot(x="class", y="odor", data=df_encoded)
plt.title("Boxplot ознаки 'odor' відносно класу")
plt.savefig("boxplot_odor.png")

# ===============================
# 4. Нормалізація
# ===============================
X = df_encoded.drop("class", axis=1)
y = df_encoded["class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ===============================
# 5. Моделі
# ===============================

# kNN
best_k = 0
best_score = 0
for k in range(3, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
print(f"🔹 kNN найкращий k={best_k}, accuracy={best_score:.4f}")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# SVM з GridSearch
param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1], "kernel": ["rbf"]}
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)
svm_best = grid.best_estimator_
print("🔹 SVM найкращі параметри:", grid.best_params_)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# AdaBoost
ab = AdaBoostClassifier(random_state=42)
ab.fit(X_train, y_train)

# ===============================
# 6. Оцінка моделей
# ===============================
models = {
    "kNN": knn_best,
    "Decision Tree": dt,
    "SVM": svm_best,
    "Random Forest": rf,
    "AdaBoost": ab
}

for name, model in models.items():
    print(f"\n=== {name} ===")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"cm_{name.replace(' ', '_')}.png")
