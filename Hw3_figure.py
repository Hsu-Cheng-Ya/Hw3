import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve


# STEP 1: 讀取資料
df = pd.read_csv("HW3_preprocessed.csv")
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# STEP 2: 特徵與標籤
X = df.drop(columns=['mortality', 'subject_id', 'stay_id', 'hadm_id'])
y = df['mortality']

# STEP 3: 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# STEP 4: 訓練模型
model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# STEP 5: 找最佳 Threshold（Youden）
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
            "Model": "Logistic Regression",
            "Threshold": round(threshold, 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "Sensitivity (TPR)": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "Youden Index": round(youden, 4)
        }

# STEP 6: 顯示結果
print("✅ 最佳 Logistic Regression 模型結果：")
for k, v in best_result.items():
    print(f"{k}: {v}")

# STEP 7: 儲存結果
pd.DataFrame([best_result]).to_csv("logistic_best_threshold_result.csv", index=False)

# === 📈 額外圖表 ===

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob):.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png")
plt.show()

# Confusion Matrix（用最佳 threshold）
y_pred_opt = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f'Confusion Matrix (Threshold = {best_threshold:.2f})')
plt.savefig("confusion_matrix.png")
plt.show()

# Calibration Plot
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.savefig("calibration_curve.png")
plt.show()
