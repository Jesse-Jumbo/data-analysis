import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 設置字體
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 讀取數據
file_path = "updated_keyword_counts_per_line.csv"
data = pd.read_csv(file_path)
data = data.dropna()

# 顯示類別分佈
print("Class distribution before filtering:")
print(data['Target'].value_counts())

# 特徵處理
# nominal_columns = ['計罰', '總額預定', '賠償', '工期', '延遲', '心證', '逾期']
nominal_columns = ['賠償', '工期', '逾期']
data['Target'] = data['Target'].map({'punitive': 1, 'compensatory': 2, 'notdefine': 0})

# 過濾極小類別
data = data[data['Target'] != 0]
print("Class distribution after filtering:")
print(data['Target'].value_counts())


# 計算詞頻
def calculate_keyword_frequencies(data, keywords):
    return data.groupby('Target')[keywords].mean()


keyword_frequencies = calculate_keyword_frequencies(data, nominal_columns)

# 分割數據
x = data[nominal_columns]
y = data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# 處理類別不平衡
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# 機器學習模型比較
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results_pdf = "model_comparison_results.pdf"
with PdfPages(results_pdf) as pdf:
    # 繪製詞頻圖
    fig, ax = plt.subplots(figsize=(12, 8))
    keyword_frequencies.T.plot(kind='bar', ax=ax, colormap="tab10")
    ax.set_title("Average Keyword Frequency by Target Class")
    ax.set_xlabel("Keywords")
    ax.set_ylabel("Average Frequency")
    ax.legend(title="Target", labels=["Punitive", "Compensatory"])
    plt.xticks(rotation=45)
    pdf.savefig(fig)
    plt.close(fig)

    # 比較模型
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train_smote, y_train_smote)

        # 性能評估
        y_pred = model.predict(x_test)
        y_scores = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

        # 混淆矩陣
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=True)
        ax.set_title(f"Confusion Matrix ({model_name})")
        pdf.savefig(fig)
        plt.close(fig)

        # 繪製 ROC 曲線
        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=2)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f"ROC Curve ({model_name})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

print(f"圖表已成功輸出至 '{results_pdf}'")