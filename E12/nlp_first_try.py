import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

# 下载nltk的必要数据
download('punkt')
download('stopwords')

# 文件路径
train_file = '../train_40.tsv'
dev_file = '../dev_40.tsv'

# 读取数据
train_data = pd.read_csv(train_file, sep='\t', header=None, names=['index', 'question', 'sentence', 'label'],
                         on_bad_lines='skip')
dev_data = pd.read_csv(dev_file, sep='\t', header=None, names=['index', 'question', 'sentence', 'label'],
                       on_bad_lines='skip')


# 数据预处理
def preprocess_text(text):
    # 转小写
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词和标点符号
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


train_data['question'] = train_data['question'].apply(preprocess_text)
train_data['sentence'] = train_data['sentence'].apply(preprocess_text)
dev_data['question'] = dev_data['question'].apply(preprocess_text)
dev_data['sentence'] = dev_data['sentence'].apply(preprocess_text)


# 词嵌入
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


glove_file = 'E:\glove.6B\glove.6B.50d.txt'
embeddings = load_glove_embeddings(glove_file)
embedding_dim = 50


# 创建词汇表和嵌入矩阵
def create_embedding_matrix(word_index, embeddings, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# 构建词汇表
from collections import Counter


def build_vocab(texts):
    vocab = Counter()
    for text in texts:
        vocab.update(text)
    word_index = {word: i + 1 for i, (word, _) in enumerate(vocab.items())}
    return word_index


texts = train_data['question'].tolist() + train_data['sentence'].tolist() + dev_data['question'].tolist() + dev_data[
    'sentence'].tolist()
word_index = build_vocab(texts)
embedding_matrix = create_embedding_matrix(word_index, embeddings, embedding_dim)


# 数据对齐与数据集
class QNLIDataset(Dataset):
    def __init__(self, data, word_index, max_len):
        self.data = data
        self.word_index = word_index
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['question']
        sentence = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']

        question_seq = [self.word_index.get(word, 0) for word in question]
        sentence_seq = [self.word_index.get(word, 0) for word in sentence]

        question_seq = self.pad_sequence(question_seq)
        sentence_seq = self.pad_sequence(sentence_seq)

        label = 1 if label == 'entailment' else 0

        return torch.tensor(question_seq), torch.tensor(sentence_seq), torch.tensor(label)

    def pad_sequence(self, seq):
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return seq


max_len = 50
batch_size = 32

train_dataset = QNLIDataset(train_data, word_index, max_len)
dev_dataset = QNLIDataset(dev_data, word_index, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)


# 构建模型
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, num_layers, dropout):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, question, sentence):
        question = self.embedding(question)
        sentence = self.embedding(sentence)

        question, _ = self.bilstm(question)
        sentence, _ = self.bilstm(sentence)

        question = torch.mean(question, dim=1)
        sentence = torch.mean(sentence, dim=1)

        combined = torch.cat((question, sentence), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output


# 超参数
hidden_dim = 128
output_dim = 2
num_layers = 2
dropout = 0.5

# 模型实例化
model = BiLSTMModel(embedding_matrix, hidden_dim, output_dim, num_layers, dropout)
model = model.cuda() if torch.cuda.is_available() else model

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练与评估函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for question, sentence, labels in train_loader:
        question, sentence, labels = question.to(device), sentence.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(question, sentence)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, dev_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for question, sentence, labels in dev_loader:
            question, sentence, labels = question.to(device), sentence.to(device), labels.to(device)

            outputs = model(question, sentence)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(dev_loader.dataset)
    return total_loss / len(dev_loader), accuracy


# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, dev_loader, criterion, device)

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
