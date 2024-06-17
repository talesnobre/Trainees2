import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from joblib import dump, load

# Load data
df = pd.read_csv("Projeto Final/data/quest.csv", encoding='ISO-8859-1')

# Mapping personality types to integers
mapa = {'ESFJ': 0, 'ESFP': 1, 'ESTJ': 2, 'ESTP': 3, 'ENFJ': 4, 'ENFP': 5,
        'ENTJ': 6, 'ENTP': 7, 'ISFJ': 8, 'ISFP': 9, 'ISTJ': 10, 'ISTP': 11,
        'INFJ': 12, 'INFP': 13, 'INTJ': 14, 'INTP': 15}
df['Personalidade'] = df['Personalidade'].replace(mapa)

# Separate the features and target
X = torch.tensor(df.drop('Personalidade', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Personalidade'].values, dtype=torch.long)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Data loaders
loaders = {
    'train': DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True),
    'test': DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)
}

# Define the model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.linear_stack(x)

# Initialize the model with correct dimensions
model = NeuralNet(X.shape[1], 128, 16)  # Note: 16 classes, not 15
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# Training loop with progress bar
epochs = 200
progress = tqdm(total=epochs * len(loaders['train']), desc='Training')

for epoch in range(1, epochs + 1):
    model.train()
    for data, target in loaders['train']:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        progress.update(1)

    # Evaluation every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loaders['test']:
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loaders['test'].dataset)
        accuracy = 100. * correct / len(loaders['test'].dataset)
        test_dataset_length = len(loaders['test'].dataset)
        print(f'Epoch {epoch}: Test set - Average loss: {test_loss:.4f}, Accuracy: {correct}/{test_dataset_length} ({accuracy:.0f}%)')

dump(model, 'rede-neural.joblib')

classifier = load('rede-neural.joblib')
progress.close()

