# Time-Series Prediction Using LSTM in PyTorch

This repository contains a project for predicting time-series data using Long Short-Term Memory (LSTM) networks implemented in PyTorch. The dataset is preprocessed to ensure a robust time-series model that predicts future values based on historical data.

## Features

- **Data Handling**: Use of pydrive to fetch and preprocess data stored on Google Drive.
- **Data Preprocessing**:
  - Interpolation for missing values.
  - Standardization of features using StandardScaler.
- **Deep Learning**:
  - Custom LSTM model for time-series prediction.
  - Implementation of backpropagation and training loop.
- **Visualization**:
  - Matplotlib for plotting training loss and predictions.
- **Performance Evaluation**:
  - Root Mean Square Error (RMSE) and Mean Square Error (MSE).
pip install -U PyDrive numpy pandas matplotlib seaborn torch statsmodels
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Data retrieval and preprocessing
def preprocessing(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y')
    dataframe.set_index("Date", inplace=True)
    return dataframe
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 40)
        self.fc = nn.Linear(40, num_classes)
        self.silu = nn.SiLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, _ = self.lstm(x, (h_0, c_0))
        hn = output[:, -1, :]
        out = self.silu(self.fc_1(hn))
        out = self.fc(out)
        return out
# Hyperparameters
num_epochs = 900
learning_rate = 0.001
input_size = 15
hidden_size = 30
num_layers = 1
num_classes = 1

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    outputs = lstm1(X_train_tensors_final)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train_tensors)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
from statsmodels.tools.eval_measures import rmse, mse

def mse_rmse(test_variable, prediction_variable):
    print(f'RMSE: {rmse(test_variable, prediction_variable)}')
    print(f'MSE: {mse(test_variable, prediction_variable)}')
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
pip install -r requirements.txt
.
├── cherat-final-dataset.csv      # Dataset file
├── lstm_time_series.py           # Main script
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
