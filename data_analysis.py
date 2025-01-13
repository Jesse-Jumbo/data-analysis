import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# 設置字體
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 讀取數據
file_path = "updated_keyword_counts_per_line.csv"
data = pd.read_csv(file_path)
data = data.dropna()

# 顯示類別分佈
print("Class distribution before SMOTE:")
print(data['Target'].value_counts())

# 特徵處理
nominal_columns = ['計罰', '總額預定', '賠償', '工期', '延遲', '心證', '逾期']
data = pd.get_dummies(data, columns=nominal_columns)
data['Target'] = data['Target'].map({'punitive': 1, 'compensatory': 2, 'notdefine': 0})

# 過濾極小類別
data = data[data['Target'] != 0]
print("Class distribution after filtering:")
print(data['Target'].value_counts())

# 分割數據
y = data["Target"]
x = data.drop(["Target", "id"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# 確保數據格式正確
x_train = x_train.astype(float)
x_test = x_test.astype(float)

# 處理類別不平衡
smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
x_train, y_train = smote.fit_resample(x_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train).value_counts())

# 訓練隨機森林模型
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_leaf': [1, 3],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='f1_weighted', verbose=1)
grid_search.fit(x_train, y_train)

# 最佳模型
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# 性能評估
y_pred = best_rf.predict(x_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 圖表輸出
results_pdf = "analysis_results.pdf"
with PdfPages(results_pdf) as pdf:
    # 類別分佈圖
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax, color="b")  # 使用單一顏色
    ax.set_title("Class Distribution Before SMOTE")
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.countplot(x=y_train, ax=ax, color="g")  # 使用單一顏色
    ax.set_title("Class Distribution After SMOTE")
    pdf.savefig(fig)
    plt.close(fig)

    # 混淆矩陣
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(best_rf, x_test, y_test, ax=ax)
    ax.set_title("Confusion Matrix")
    pdf.savefig(fig)
    plt.close(fig)

    # ROC曲線
    y_test_proba = best_rf.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba, pos_label=2)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

print(f"圖表已成功匯出至 '{results_pdf}'")
