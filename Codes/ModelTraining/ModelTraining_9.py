import os
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 📂 Paths
FEATURES_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Features"
MODELS_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Models\Test-9"
RESULTS_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 🔹 Load DenseNet features
X = np.load(os.path.join(FEATURES_DIR, "DenseNet_features_8.npy"))
y = np.load(os.path.join(FEATURES_DIR, "DenseNet_labels_8.npy"))

print(f"✅ Loaded features: {X.shape}, labels: {y.shape}")

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# ---------------- SVM ----------------
print("\n🔹 Training SVM...")
svm = SVR(C=10, kernel="rbf", gamma="scale")  # tuned params
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

mae_svm = mean_absolute_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

joblib.dump(svm, os.path.join(MODELS_DIR, "SVM_DenseNet.pkl"))
print(f"✅ SVM saved. MAE={mae_svm:.2f}, R²={r2_svm:.2f}")
results.append(("SVM", mae_svm, r2_svm))

# ---------------- RandomForest ----------------
print("\n🔹 Training RandomForest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

joblib.dump(rf, os.path.join(MODELS_DIR, "RandomForest_DenseNet.pkl"))
print(f"✅ RandomForest saved. MAE={mae_rf:.2f}, R²={r2_rf:.2f}")
results.append(("RandomForest", mae_rf, r2_rf))

# ---------------- DecisionTree ----------------
print("\n🔹 Training DecisionTree...")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

joblib.dump(dt, os.path.join(MODELS_DIR, "DecisionTree_DenseNet.pkl"))
print(f"✅ DecisionTree saved. MAE={mae_dt:.2f}, R²={r2_dt:.2f}")
results.append(("DecisionTree", mae_dt, r2_dt))

# ---------------- Save Results ----------------
results_path = os.path.join(RESULTS_DIR, "DenseNet_results_9.txt")
with open(results_path, "w") as f:
    for model_name, mae, r2 in results:
        f.write(f"{model_name} -> MAE: {mae:.2f}, R²: {r2:.2f}\n")

print(f"\n📂 Results saved in {results_path}")
