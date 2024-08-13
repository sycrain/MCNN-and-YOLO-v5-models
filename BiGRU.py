import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 自定义数据集类
class AminoAcidDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.loadtxt(self.file_paths[idx]).astype(np.float32)
        return torch.tensor(data).unsqueeze(0), torch.tensor(self.labels[idx])

# BiGRU 模型
class BiGRU(nn.Module):
    def __init__(self, num_classes=10):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(128 * 2, 128)  # 128*2 因为 BiGRU 是双向的
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.relu(self.fc1(x[:, -1, :]))  # 使用 GRU 的最后一个隐藏状态
        x = self.fc2(x)
        return x

# 数据集路径
data_dir = 'dataset'

# 加载数据
file_paths = []
labels = []
label_names = os.listdir(data_dir)
for label_index, label in enumerate(label_names):
    label_dir = os.path.join(data_dir, label)
    for txt_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, txt_file)
        file_paths.append(file_path)
        labels.append(label_index)

# 拆分数据集为训练集和测试集
train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.3, random_state=42)

# 创建数据加载器
train_dataset = AminoAcidDataset(train_paths, train_labels)
test_dataset = AminoAcidDataset(test_paths, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiGRU(num_classes=len(label_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
start_train_time = time.time()
model.train()
for epoch in range(10):  # 可以调整 epoch 数量
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.squeeze(1).to(device), labels.to(device)  # 调整数据维度

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

end_train_time = time.time()

# 测试模型
model.eval()
start_test_time = time.time()
all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.squeeze(1).to(device), labels.to(device)  # 调整数据维度

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

end_test_time = time.time()

# 计算评估指标
conf_matrix = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
recall_per_class = recall_score(all_labels, all_preds, average=None)
f1_per_class = f1_score(all_labels, all_preds, average=None)
precision_per_class = precision_score(all_labels, all_preds, average=None)
accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# 输出每种氨基酸的评估指标
for i, label_name in enumerate(label_names):
    print(f"Results for {label_name}:")
    print(f"  Accuracy: {accuracy_per_class[i]:.4f}")
    print(f"  Recall: {recall_per_class[i]:.4f}")
    print(f"  F1 Score: {f1_per_class[i]:.4f}")
    print(f"  Precision: {precision_per_class[i]:.4f}")
    print()

# 输出总体训练时间和测试时间
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Training Time: {end_train_time - start_train_time:.4f} seconds")
print(f"Test Time: {end_test_time - start_test_time:.4f} seconds")

# 绘制混淆矩阵并保存为图片
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xticks(rotation=45)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
