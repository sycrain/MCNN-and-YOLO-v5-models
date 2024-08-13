import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time


def load_data(data_dir):
    file_paths = []
    labels = []
    label_names = os.listdir(data_dir)

    for label_index, label in enumerate(label_names):
        label_dir = os.path.join(data_dir, label)
        for txt_file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, txt_file)
            file_paths.append(file_path)
            labels.append(label_index)

    return file_paths, labels, label_names


def read_txt_file(file_path):
    # 使用制表符作为分隔符
    data = np.loadtxt(file_path, delimiter='\t').astype(np.float32)
    return data.flatten()


def main():
    # 数据集路径
    data_dir = 'dataset'

    # 加载数据
    file_paths, labels, label_names = load_data(data_dir)
    data = np.array([read_txt_file(fp) for fp in file_paths])
    labels = np.array(labels)

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练SVM模型
    start_train_time = time.time()
    model = SVC(kernel='linear')  # 使用线性核，可以根据需要选择其他核函数
    model.fit(X_train, y_train)
    end_train_time = time.time()

    # 预测
    start_test_time = time.time()
    y_pred = model.predict(X_test)
    end_test_time = time.time()

    # 计算评估指标
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_per_class = precision_score(y_test, y_pred, average=None)
    accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # 输出每种氨基酸的评估指标
    for i, label_name in enumerate(label_names):
        print(f"Results for {label_name}:")
        print(f"  Accuracy: {accuracy_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1 Score: {f1_per_class[i]:.4f}")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print()

    print(f"Training time: {end_train_time - start_train_time:.2f} seconds")
    print(f"Test time: {end_test_time - start_test_time:.2f} seconds")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xticks(rotation=45)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # 保存为图片
    plt.show()


if __name__ == "__main__":
    main()
