from ViT import ViT_model
from torchvision.datasets import ImageFolder
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


# 设置设备为GPU（如果有可用的）否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 将图像调整为224x224大小
    transforms.ToTensor(),              # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # 使用均值和标准差进行标准化
])


# 定义训练集和测试集的路径
train_dir = './dogvscat/training_set'
test_dir = './dogvscat/test_set'

# 使用ImageFolder加载数据集，并应用预处理变换
train_datasets = ImageFolder(train_dir, transform)
test_datasets = ImageFolder(test_dir, transform)

batch_size = 128        # 定义批量大小

# 创建数据加载器，用于批量加载数据
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

model = ViT_model()
model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


n_epochs = 10
best_acc = 0


for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    # 创建进度条，用于显示训练进度
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch{epoch+1}/{n_epochs}")

    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = loss_function(output, labels)    # 计算损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()     # 累加损失

        pbar.set_postfix({'loss': running_loss/(i+1)})      # 更新进度条的后缀信息，显示当前损失

    model.eval()
    # 初始化正确预测和总数变量
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, idx = torch.max(output.data, -1)
            total += labels.size(0)
            correct += (idx == labels).sum().item()

    acc = correct / total
    print('Epoch {}: Accuracy on the test set: {}'.format(epoch+1, acc))

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_vit_model.pth')

print('finish')