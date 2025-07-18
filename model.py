# model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
import mlflow

def model(features, test_features, encoding='ohe', n_folds=5):
    mlflow.set_experiment("LightGBM_HomeCredit")

    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    labels = features['TARGET']

    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        features, test_features = features.align(test_features, join='inner', axis=1)
        cat_indices = 'auto'
    elif encoding == 'le':
        label_encoder = LabelEncoder()
        cat_indices = []
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(features[col].astype(str))
                test_features[col] = label_encoder.transform(test_features[col].astype(str))
                cat_indices.append(i)
    else:
        raise ValueError("Encoding must be 'ohe' or 'le'")

    print('Training Data Shape:', features.shape)
    print('Testing Data Shape:', test_features.shape)

    feature_names = list(features.columns)
    features = np.array(features)
    test_features = np.array(test_features)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    valid_scores = []
    train_scores = []

    with mlflow.start_run(run_name="LightGBM_CV"):
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("encoding", encoding)

        for fold, (train_indices, valid_indices) in enumerate(k_fold.split(features)):
            print(f"\n🔁 Fold {fold + 1}/{n_folds}")
            X_train, y_train = features[train_indices], labels.iloc[train_indices]
            X_valid, y_valid = features[valid_indices], labels.iloc[valid_indices]

            clf = lgb.LGBMClassifier(
                n_estimators=10000,
                objective='binary',
                class_weight='balanced',
                learning_rate=0.05,
                reg_alpha=0.1,
                reg_lambda=0.1,
                subsample=0.8,
                n_jobs=-1,
                random_state=50
            )

            clf.fit(
                X_train, y_train,
                eval_metric='auc',
                eval_set=[(X_valid, y_valid), (X_train, y_train)],
                eval_names=['valid', 'train'],
                categorical_feature=cat_indices,
                callbacks=[
                    early_stopping(stopping_rounds=100),
                    log_evaluation(period=200)
                ]
            )

            best_iteration = clf.best_iteration_
            feature_importance_values += clf.feature_importances_ / n_folds
            test_predictions += clf.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / n_folds
            out_of_fold[valid_indices] = clf.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]

            auc_valid = roc_auc_score(y_valid, clf.predict_proba(X_valid, num_iteration=best_iteration)[:, 1])
            auc_train = roc_auc_score(y_train, clf.predict_proba(X_train, num_iteration=best_iteration)[:, 1])

            valid_scores.append(auc_valid)
            train_scores.append(auc_train)

            mlflow.log_metric(f"fold{fold+1}_valid_auc", auc_valid)
            mlflow.log_metric(f"fold{fold+1}_train_auc", auc_train)

            del clf, X_train, X_valid, y_train, y_valid
            gc.collect()

        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        valid_auc = roc_auc_score(labels, out_of_fold)
        overall_train = np.mean(train_scores)

        fold_names = list(range(n_folds)) + ['overall']
        train_scores.append(overall_train)
        valid_scores.append(valid_auc)
        metrics = pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores})

        metrics.to_csv("cv_metrics.csv", index=False)
        feature_importances.to_csv("feature_importances.csv", index=False)
        submission.to_csv("submission.csv", index=False)

        mlflow.log_metric("overall_valid_auc", valid_auc)
        mlflow.log_metric("overall_train_auc", overall_train)
        mlflow.log_artifact("cv_metrics.csv")
        mlflow.log_artifact("feature_importances.csv")
        mlflow.log_artifact("submission.csv")

    return submission, feature_importances, metrics, feature_names, features, labels
