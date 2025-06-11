import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# STEP 1: 讀取資料
df = pd.read_csv("HW3_preprocessed.csv")  # 確保與你的檔案路徑一致

# 將 gender 轉換為數值 0（F）和 1（M）
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# STEP 2: 特徵與標籤分開
X = df.drop(columns=['mortality', 'subject_id', 'stay_id', 'hadm_id'])
y = df['mortality']

# STEP 3: 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# STEP 4: 訓練 Logistic Regression 模型
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]  # 機率值

# STEP 5: 掃描最佳 Threshold（最大 Youden index）
best_threshold = 0
best_youden = -1
best_result = {}

for threshold in np.arange(0, 1.01, 0.01):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    if cm.shape != (2, 2):
        continue  # 避免不平衡分類造成錯誤
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    
    if youden > best_youden:
        best_youden = youden
        best_threshold = threshold
        best_result = {
            "Model": "Logistic Regression",
            "Threshold": round(threshold, 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "Sensitivity (TPR)": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "Youden Index": round(youden, 4)
        }

# STEP 6: 顯示最佳結果
print("✅ 最佳 Logistic Regression 模型結果：")
for k, v in best_result.items():
    print(f"{k}: {v}")

# STEP 7: 可選：儲存成 CSV
pd.DataFrame([best_result]).to_csv("logistic_best_threshold_result.csv", index=False)
