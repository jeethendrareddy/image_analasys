# feature_selection_and_eval.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# CONFIG
FEATURES_CSV = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Features\dal_features_3.csv"
RANDOM_STATE = 42
OUT_DIR = r"D:\PyCharm Community Edition 2024.3.5\PROJECTS\Arhar_Khesari_Dal\Features\dal_reduced_feature_1.csv"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(FEATURES_CSV)
print("Loaded features:", df.shape)

# Split
X = df.drop(columns=['Label','ImageName'], errors='ignore')
y = df['Label'].values
groups = df['ImageName'].values

feature_names = X.columns.tolist()
X = X.values

# 0) quick sanity: remove near-constant features
vt = VarianceThreshold(threshold=1e-5)
X_vt = vt.fit_transform(X)
kept_mask = vt.get_support()
kept_names = [n for n,keep in zip(feature_names, kept_mask) if keep]
print(f"VarianceThreshold kept {len(kept_names)} features (removed {len(feature_names)-len(kept_names)})")

# 1) remove highly correlated features (pairwise)
# build dataframe of the kept features for correlation calc
X_df = pd.DataFrame(X_vt, columns=kept_names)
corr = X_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
print(f"Dropping {len(to_drop)} highly correlated features (corr>0.95)")
X_uncorr_df = X_df.drop(columns=to_drop)
uncorr_names = X_uncorr_df.columns.tolist()
X_uncorr = X_uncorr_df.values

# Use GroupKFold for CV
gkf = GroupKFold(n_splits=5)

# Helper to evaluate pipeline
def eval_pipeline(pipeline, X_in, y, groups, scoring='accuracy'):
    if scoring == 'accuracy':
        scorer = make_scorer(accuracy_score)
    else:
        scorer = make_scorer(r2_score)
    scores = cross_val_score(pipeline, X_in, y, groups=groups, cv=gkf, scoring=scorer, n_jobs=-1)
    return scores

results = []

# Baseline: RandomForest on uncorrelated features (no feature selection)
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
scores = eval_pipeline(rf, X_uncorr, y, groups, scoring='accuracy')
print("RF baseline accuracy (5-fold grouped): mean %.3f ± %.3f" % (scores.mean(), scores.std()))
results.append({'method':'RF_baseline','n_features':X_uncorr.shape[1],'acc_mean':scores.mean(),'acc_std':scores.std()})

# 2) Univariate selection (SelectKBest) try multiple k
ks = [10, 20, 30, 40]
for k in ks:
    sk = SelectKBest(score_func=f_classif, k=min(k, X_uncorr.shape[1]))
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([('sel', sk), ('clf', clf)])
    scores = eval_pipeline(pipe, X_uncorr, y, groups, scoring='accuracy')
    print(f"SelectKBest k={k} -> mean acc {scores.mean():.3f} ± {scores.std():.3f}")
    # store selected names for best later
    pipe.fit(X_uncorr, y)
    selected_idx = sk.get_support(indices=True)
    sel_names = [uncorr_names[i] for i in selected_idx]
    # Save reduced dataframe with Label & ImageName
    reduced_df = df[["ImageName", "Label"] + sel_names]
    reduced_df.to_csv(os.path.join(OUT_DIR, f"SelectKBest_k{k}_features.csv"), index=False)
    results.append({'method':f'SelectKBest_k{k}','n_features':len(sel_names),'acc_mean':scores.mean(),'acc_std':scores.std()})

# 3) RFE with RandomForest (recursive selection) — slower, but often effective
# choose target features to try
rfe_targets = [10, 20, 30]
estimator_for_rfe = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
for target in rfe_targets:
    print(f"Running RFE -> target {target} features (this may take a bit)")
    rfe = RFE(estimator=estimator_for_rfe, n_features_to_select=min(target, X_uncorr.shape[1]), step=0.2)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([('rfe', rfe), ('clf', clf)])
    scores = eval_pipeline(pipe, X_uncorr, y, groups, scoring='accuracy')
    print(f"RFE target={target} -> mean acc {scores.mean():.3f} ± {scores.std():.3f}")
    # save selected features
    pipe.fit(X_uncorr, y)
    sel_idx = rfe.get_support(indices=True)
    sel_names = [uncorr_names[i] for i in sel_idx]
    # Save reduced dataframe with Label & ImageName
    reduced_df = df[["ImageName", "Label"] + sel_names]
    reduced_df.to_csv(os.path.join(OUT_DIR, f"RFE_{target}_features.csv"), index=False)
    results.append({'method':f'RFE_{target}','n_features':len(sel_names),'acc_mean':scores.mean(),'acc_std':scores.std()})

# 4) PCA -> use components to retain 0.90/0.95 variance
for var in [0.90, 0.95]:
    pca = PCA(n_components=var, svd_solver='full', random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([('sc', StandardScaler()), ('pca', pca), ('clf', clf)])  # scale before PCA
    scores = eval_pipeline(pipe, X_uncorr, y, groups, scoring='accuracy')
    # estimate n_components after fitting on whole data
    pca.fit(StandardScaler().fit_transform(X_uncorr))
    n_comp = pca.n_components_
    print(f"PCA retain {var:.2f} -> n_comp={n_comp}, mean acc {scores.mean():.3f} ± {scores.std():.3f}")
    results.append({'method':f'PCA_{int(var*100)}','n_features':n_comp,'acc_mean':scores.mean(),'acc_std':scores.std()})

# 5) Save results summary
res_df = pd.DataFrame(results).sort_values('acc_mean', ascending=False)
res_df.to_csv(os.path.join(OUT_DIR, "feature_selection_summary.csv"), index=False)
print("Saved summary to", os.path.join(OUT_DIR, "feature_selection_summary.csv"))
print(res_df)
