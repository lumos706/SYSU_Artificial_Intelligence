import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd


def preprocess_data(file_path, max_length, embedding_model):
    """
    函数功能：一体化数据处理。读取数据，对数据进行预处理，词嵌入和数据对齐
    :param file_path: 文件路径
    :param max_length: 句子最大长度 --> 对于整个数据集，我们将所有句子都填充或截断到相同的长度60，再大可能会使数组过大而报错
    :param embedding_model: 词嵌入模型
    """
    questions = []
    hypotheses = []
    labels = []
    embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

    df = pd.read_csv(file_path, sep='\t', header=0, on_bad_lines='skip')

    for _, row in df.iterrows():
        question = nltk.word_tokenize(row['question'])[:max_length]
        hypothesis = nltk.word_tokenize(row['sentence'])[:max_length]

        question_vectors = [embedding_model[word] if word in embedding_model else np.zeros(embedding_dim) for word in
                            question]
        hypothesis_vectors = [embedding_model[word] if word in embedding_model else np.zeros(embedding_dim) for word in
                              hypothesis]

        while len(question_vectors) < max_length:
            question_vectors.append(np.zeros_like(question_vectors[0]))
        while len(hypothesis_vectors) < max_length:
            hypothesis_vectors.append(np.zeros_like(hypothesis_vectors[0]))

        questions.append(np.array(question_vectors, dtype='float32'))
        hypotheses.append(np.array(hypothesis_vectors, dtype='float32'))
        labels.append(int(row['label'] == 'entailment'))

    return questions, hypotheses, labels


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, question, hypothesis):
        h0 = torch.zeros(self.lstm.num_layers * 2, question.size(0), self.lstm.hidden_size).to(question.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, question.size(0), self.lstm.hidden_size).to(question.device)

        h_question, _ = self.lstm(question, (h0, c0))
        h_hypothesis, _ = self.lstm(hypothesis, (h0, c0))

        h_question = h_question[:, -1, :]
        h_hypothesis = h_hypothesis[:, -1, :]

        combined = torch.cat((h_question, h_hypothesis), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output


def train(model, data_loader, criterion, optimizer):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for (question, hypothesis), labels in data_loader:
        question = question
        hypothesis = hypothesis
        labels = labels
        optimizer.zero_grad()
        outputs = model(question, hypothesis)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * question.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for (question, hypothesis), labels in data_loader:
            question = question
            hypothesis = hypothesis
            labels = labels
            outputs = model(question, hypothesis)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * question.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_qnli_model(train_file, valid_file, max_length, batch_size, embedding_model, num_epochs):
    embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

    train_questions, train_hypotheses, train_labels = preprocess_data(train_file, max_length, embedding_model)
    valid_questions, valid_hypotheses, valid_labels = preprocess_data(valid_file, max_length, embedding_model)

    train_dataset = list(zip(zip(train_questions, train_hypotheses), train_labels))
    valid_dataset = list(zip(zip(valid_questions, valid_hypotheses), valid_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(valid_dataset))

    input_dim = embedding_dim
    hidden_dim = 128
    n_layers = 2
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, output_dim, n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    loss = []
    accuracy = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        end_time = time.time()

        print('Epoch:', epoch + 1)
        print('Training Loss:', train_loss, 'Training Accuracy:', train_acc)
        print('Validation Loss:', valid_loss, 'Validation Accuracy:', valid_acc)
        print('Time per epoch:', end_time - start_time, 'seconds')

        loss.append((train_loss, valid_loss))
        accuracy.append((train_acc, valid_acc))

    return loss, accuracy


def plot_curves(train_loss, train_acc, valid_loss, valid_acc):
    epochs = range(1, len(train_loss) + 1)

    # 绘制训练集和验证集的损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, valid_loss, 'r-', label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练集和验证集的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, valid_acc, 'r-', label='Valid Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


def main():
    # 定义训练集和验证集文件路径
    train_file = '../train_40.tsv'
    valid_file = '../dev_40.tsv'

    # 定义句子最大长度和批次大小
    max_length = 30
    batch_size = 32
    num_epochs = 10

    # 加载GloVe词向量模型
    embedding_model = {}
    with open('E:\glove.6B\glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_model[word] = vector
    embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

    loss, accuracy = train_qnli_model(train_file, valid_file, max_length, batch_size, embedding_model, num_epochs)
    train_loss, valid_loss = zip(*loss)
    train_acc, valid_acc = zip(*accuracy)

    plot_curves(train_loss, train_acc, valid_loss, valid_acc)


if __name__ == "__main__":
    main()
