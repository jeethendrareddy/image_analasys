import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import FEATURES_RESNET_7, MODELS_TEST_8_DIR, RESULTS_DIR, RANDOM_STATE

os.makedirs(MODELS_TEST_8_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Load features & labels
# ----------------------------
X = np.load(FEATURES_RESNET_7['features'])
y = np.load(FEATURES_RESNET_7['labels'])

print("✅ Loaded data:", X.shape, y.shape)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ----------------------------
# Define models
# ----------------------------
models = {
    "SVM": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=50, gamma="scale")),
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
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
    model_path = os.path.join(MODELS_TEST_8_DIR, f"{name}_model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ {name} saved at {model_path}")
    print(f"📊 {name} -> MAE: {mae:.2f}, R²: {r2:.2f}")

    results.append(f"{name}: MAE={mae:.2f}, R²={r2:.2f}")

# ----------------------------
# Save results
# ----------------------------
results_file = os.path.join(RESULTS_DIR, "model_comparison_Test-8.txt")
with open(results_file, "w") as f:
    f.write("\n".join(results))

print(f"\n📂 Results saved in {results_file}")
