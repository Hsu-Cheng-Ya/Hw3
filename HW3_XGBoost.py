import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# 讀取資料
df = pd.read_csv("HW3_preprocessed.csv")

# 編碼 gender
if df['gender'].dtype == 'object':
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# 分割資料
X = df.drop(columns=['mortality', 'subject_id', 'stay_id', 'hadm_id'])
y = df['mortality']

# 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 建立 XGBoost 模型
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# 預測機率
y_prob = model.predict_proba(X_test)[:, 1]

# 掃描最佳 threshold
best_threshold = 0
best_youden = -1
best_result = {}

for threshold in np.arange(0, 1.01, 0.01):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    if cm.shape != (2, 2):
        continue

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1

    if youden > best_youden:
        best_youden = youden
        best_threshold = threshold
        best_result = {
            "Model": "XGBoost",
            "Threshold": round(threshold, 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "Sensitivity (TPR)": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "Youden Index": round(youden, 4)
        }

# 顯示結果
print("✅ 最佳 XGBoost 模型結果：")
for k, v in best_result.items():
    print(f"{k}: {v}")

# 儲存
pd.DataFrame([best_result]).to_csv("xgboost_best_result.csv", index=False)

