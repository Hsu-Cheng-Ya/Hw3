# 套件匯入
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. 讀取資料
file_path = "Hw2_final_feature.csv"  # 改成你本機的路徑
df = pd.read_csv(file_path)

# 2. 排除不需要處理的欄位
exclude_cols = ['age', 'mortality', 'vasopressin_use']  # 這些欄位不需處理
features_to_process = [col for col in df.select_dtypes(include=['float64', 'int64']).columns
                       if col not in exclude_cols]

# 3. capped outlier 處理函式
def cap_outliers(df, cols):
    df_capped = df.copy()
    for col in cols:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_capped[col] = np.clip(df_capped[col], lower, upper)
    return df_capped

# 4. 執行 capped outlier
df[features_to_process] = cap_outliers(df[features_to_process], features_to_process)

# 5. multivariate imputation 處理缺值
imputer = IterativeImputer(random_state=0)
df[features_to_process] = imputer.fit_transform(df[features_to_process])

# 6. 最終檢查缺值
missing = df.isnull().sum()
print("缺值檢查：")
print(missing[missing > 0])

# 7. 儲存處理後的資料（可選）
df.to_csv("HW3_preprocessed.csv", index=False)
print("✅ 資料處理完成，結果已儲存為 'HW3_preprocessed.csv'")
