import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# Paths
# ----------------------------
FEATURES_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Features"
MODELS_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Models\Test-8"
RESULTS_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Load features & labels
# ----------------------------
X = np.load(os.path.join(FEATURES_DIR, "dal_features_7.npy"))
y = np.load(os.path.join(FEATURES_DIR, "dal_labels_7.npy"))

print("✅ Loaded data:", X.shape, y.shape)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Define models
# ----------------------------
models = {
    "SVM": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=50, gamma="scale")),
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
}

results = []

# ----------------------------
# Train & evaluate
# ----------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ {name} saved at {model_path}")
    print(f"📊 {name} -> MAE: {mae:.2f}, R²: {r2:.2f}")

    results.append(f"{name}: MAE={mae:.2f}, R²={r2:.2f}")

# ----------------------------
# Save results
# ----------------------------
results_file = os.path.join(RESULTS_DIR, "model_comparision_Test-8.txt")
with open(results_file, "w") as f:
    f.write("\n".join(results))

print(f"\n📂 Results saved in {results_file}")
