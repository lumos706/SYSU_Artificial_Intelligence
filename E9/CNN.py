import warnings
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS = 30  # 总共训练批次
BATCH_SIZE = 32  # 每个批次的大小
learning_rate = 0.001  # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using  {DEVICE}  device')

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图片大小
    transforms.ToTensor()
])

# 导入图片数据集
train_data = datasets.ImageFolder(
    root="../cnn_data/train",
    transform=transform  # 使用定义的transform变量
)

test_data = datasets.ImageFolder(
    root="../cnn_data/test",
    transform=transform  # 使用定义的transform变量
)

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 打乱数据
    drop_last=False)  # 不丢弃最后一个不完整的batch

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=False,  # 不打乱数据
    drop_last=False)  # 不丢弃最后一个不完整的batch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 卷积层，3个输入通道，16个输出通道
            torch.nn.BatchNorm2d(16),  # 批标准化
            torch.nn.ReLU(),  # 激活函数
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，缩小图片尺寸
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5)  # 添加Dropout层
        )
        # 根据中草药类型的数量更改输出层的节点数
        self.fc = torch.nn.Linear(32 * 32 * 32, 5)  # 全连接层，5个神经元

    def forward(self, x):
        x = self.layer1(x)  # 通过第一个卷积层
        x = self.layer2(x)  # 通过第二个卷积层
        x = x.view(x.size(0), -1)  # 将图片数据展平
        x = self.fc(x)  # 通过全连接层
        return x


def model_test(model, epoch):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0
    test_acc = 0
    progress_bar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc="Testing")
    with torch.no_grad():
        for i, (images, labels) in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 获取每一层的输出
            x = model.layer1(images)
            x1 = model.layer2(x)
            outputs = model(images)

            if i == 1 and epoch == 0:
                plot_images_and_layers(images, x, x1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_acc += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {'Test Loss': '{:.4f}'.format(test_loss / len(test_data_loader)),
                 'Test Accuracy': '{:.4f}'.format(test_acc / len(test_data_loader.dataset))})

    test_loss /= len(test_data_loader)
    test_acc /= len(test_data_loader.dataset)
    return test_loss, test_acc


def train():
    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # SGD优化器
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)  # RMSprop优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # SGD优化器，使用momentum
    # 定义学习率调度器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # 每10个epoch学习率乘以0.1

    train_loss_list = []  # 记录每个epoch的损失
    train_acc_list = []  # 记录每个epoch的准确率
    test_loss_list = []  # 记录每个epoch的测试损失
    test_acc_list = []  # 记录每个epoch的测试准确率

    for epoch in range(EPOCHS):
        train_loss = 0
        train_acc = 0
        # 在每个epoch开始时调用scheduler.step()
        scheduler.step()

        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                            desc="Epoch {}".format(epoch + 1))  # 进度条
        for i, (images, labels) in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            # print(labels)

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # print(outputs.data)
            train_loss += loss.item()  # 计算损失
            _, predicted = torch.max(outputs.data, 1)  # 找到预测结果
            # print(predicted)
            train_acc += (predicted == labels).sum().item()  # 计算准确率
            # print((predicted == labels).sum())
            progress_bar.set_postfix(
                {'Train Loss': '{:.4f}'.format(train_loss / len(train_data_loader)),
                 'Train Accuracy': '{:.4f}'.format(train_acc / len(train_data_loader.dataset))})
        # print(train_acc, len(train_data_loader.dataset))
        train_loss /= len(train_data_loader)
        train_acc /= len(train_data_loader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        test_loss, test_acc = model_test(model, epoch)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # print(f"\nEpoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Progress Finished")
    # 保存模型参数
    # torch.save(model.state_dict(), 'model.pth')
    return model, train_loss_list, train_acc_list, test_loss_list, test_acc_list


def plot_images_and_layers(images, x, x1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    for k in range(3):
        axs[k].imshow(images[0, k, :, :].cpu().numpy(), cmap='gray')
        axs[k].axis('off')
    plt.suptitle('Input Image')
    plt.savefig('../pics/input_image.png')

    for j, img in enumerate([x, x1]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (
        0, 2, 3, 1))  # (batch_size, channels, height, width) -> (batch_size, height, width, channels)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        num_channels = img.shape[3]  # 获取通道数
        fig, axs = plt.subplots(4, num_channels // 4, figsize=(15, 15))  # 创建一个1行，num_channels列的子图

        for m in range(4):
            for n in range(num_channels // 4):
                channel_idx = m * 4 + n
                axs[m, n].imshow(img[0, :, :, channel_idx], cmap='gray')  # 在第i个子图中显示第i个通道的图像
                axs[m, n].axis('off')  # 关闭坐标轴
        plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 设置子图之间的间隔
        plt.suptitle(f'Layer{j + 1} Output')  # 设置标题
        plt.savefig(f'../pics/layer{j + 1}_output.png')


def plot_graphs(train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss_list)), train_loss_list, color='b', label='Train Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(0, EPOCHS + 2, 2))
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Batch Size:{}'.format(BATCH_SIZE))
    plt.savefig('../pics/train_loss.png')
    plt.show()

    # 绘制训练准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_acc_list)), train_acc_list, color='r', label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(range(0, EPOCHS + 2, 2))
    plt.ylabel('Accuracy')
    # 找到最大准确率的位置
    max_acc = max(train_acc_list)
    max_acc_epoch = train_acc_list.index(max_acc)
    # 在最大准确率的位置添加标注
    plt.annotate(f'Max: {max_acc}', xy=(max_acc_epoch, max_acc), xytext=(max_acc_epoch, max_acc - 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend()
    plt.title('Batch Size:{}'.format(BATCH_SIZE))
    plt.savefig('../pics/train_accuracy.png')
    plt.show()

    # 绘制测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(test_loss_list)), test_loss_list, color='b', label='Test Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(0, EPOCHS + 2, 2))
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Batch Size:{}'.format(BATCH_SIZE))
    plt.savefig('../pics/test_loss.png')
    plt.show()

    # 绘制测试准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(test_acc_list)), test_acc_list, color='r', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(range(0, EPOCHS + 2, 2))
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Batch Size:{}'.format(BATCH_SIZE))
    plt.savefig('../pics/test_accuracy.png')
    plt.show()


def main():
    # 输出数据集的大小
    print(f"train_data_loader length: {len(train_data_loader)}")
    print(f"test_data_loader length: {len(test_data_loader)}")

    begin_time = time.time()
    model, train_loss_list, train_acc_list, test_loss_list, test_acc_list = train()
    end_time = time.time()
    print(f"Cost time: {end_time - begin_time:.2f}s")

    # 调用绘图函数
    plot_graphs(train_loss_list, train_acc_list, test_loss_list, test_acc_list)


if __name__ == '__main__':
    main()
