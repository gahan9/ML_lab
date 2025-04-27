# Data Processing Modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural net processing Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# Plotting modules
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D


# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.X = features
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_df, target_df, hidden_layers=(256, 64, 16), *args, **kwargs):
        """

        :param input_df: input data frame with the features.
        :param target_df: target data frame with labels.
        :param hidden_layers: list of neurons in each hidden layer.
        :param args:
        :param kwargs:
        """
        super(NeuralNetwork, self).__init__()
        self._device = None
        self.batch_size = kwargs.get('batch_size', 512)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.num_epochs = kwargs.get('num_epochs', 1000)
        self.output_size = kwargs.get('output_size', 1)
        self.scaler = kwargs.get('scaler', StandardScaler())
        self.output_scaler = kwargs.get('output_scaler', None)
        self.hidden_layers = hidden_layers
        self.target_df = target_df
        self.input_df = input_df
        self.model = None
        self.activation = nn.Softplus()
        self.X_train = kwargs.get('X_train', None)
        self.X_val = kwargs.get('X_val', None)
        self.y_train = kwargs.get('y_train', None)
        self.y_val = kwargs.get('y_val', None)
        self.split_data = kwargs.get('split_data', 0)
        self.layers = self.construct_layers()
        print(self)
        self.train_test_split()
        self.train_loss = 0

    def __str__(self):
        """
        String representation of the neural network model.
        :return:
        """
        _str = super().__str__()
        _str += f"\nInput Layer Size: {self.input_layer_size}"
        _str += f"\nInput Layer Size: {self.layers}"
        return _str

    @property
    def input_layer_size(self):
        return len(self.input_df.columns)

    def construct_layers(self):
        """Construct neural network layers based on the specified architecture.

        :return:
        """
        # initialize model layers
        layers = []
        prev_size = self.input_layer_size
        for idx, size in enumerate(self.hidden_layers):
            setattr(self, f"layer_{idx}", nn.Linear(prev_size, size))
            layers.append(getattr(self, f"layer_{idx}"))
            prev_size = size

        # Add output layer
        setattr(self, "output", nn.Linear(prev_size, self.output_size))
        return layers

    def forward(self, x):
        """
        Forward pass through the neural network.
        :return:
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output(x)
        return x

    @property
    def loss_function(self):
        """
        Loss function for the model.
        :return:
        """
        return nn.MSELoss()

    def model_optimizer(self):
        """
        Optimizer for the model.

        :return:
        """
        return optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

    def input_data(self):
        """
        Input data for the model.
        :return:
        """
        _df = self.X_train.copy()
        if self.scaler is not None:
            _df = self.scaler.fit_transform(_df)
        return _df

    def target_data(self):
        """
        Target data for the model.
        :return:
        """
        _df = self.y_train.copy()
        if self.output_scaler is not None:
            _df = self.scaler.fit_transform(_df)
        return _df

    def training_data(self, shuffle=True):
        """
        Training data for the model.
        :return: data.DataLoader
        """
        _input_df = self.input_data()
        _target_df = self.target_data()
        train_dataset = RegressionDataset(
            torch.tensor(_input_df, dtype=torch.float32),
            torch.tensor(_target_df.values, dtype=torch.float32).reshape(-1, 1)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
        return train_loader

    def validation_data(self, shuffle=True):
        """
        Return Validation data for the model if available.

        :return: data.DataLoader/None
        """
        if self.split_data:
            _input_df = self.scaler.fit_transform(self.X_val)
            _target_df = self.y_val.copy()
            _dataset = RegressionDataset(
                torch.tensor(_input_df, dtype=torch.float32),
                torch.tensor(_target_df.values, dtype=torch.float32).reshape(-1, 1)
            )
            _data_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=shuffle)
            return _data_loader

        return None

    def train_test_split(self):
        if self.split_data and 0 < self.split_data < 1:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.input_df,
                self.target_df,
                test_size=self.split_data,
                random_state=47
            )
        else:
            self.X_train = self.input_df
            self.X_val = None
            self.y_train = self.target_df
            self.y_val = None

    def train_model(self):
        """Train the model.
        :return:
        """
        self.train()
        self.train_loss = 0
        train_loader = self.training_data(shuffle=True)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            self.train_loss += loss.item() * inputs.size(0)

            self.train_loss = self.train_loss / len(train_loader.dataset)


    def predict_test(self, input_data):
        """Predict on test data.

        :return:
        """
        sample_input = torch.tensor(input_data, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(sample_input)
        print(f'Predicted value: {prediction.item()}')


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    train_df = pd.read_csv("./data/train.csv")
    feature_df = pd.read_csv("./data/features.csv")
    train_df = train_df.set_index(["Store", "Date"])
    training_df = train_df.merge(feature_df.set_index(['Store', 'Date']), left_index=True, right_index=True)

    cols_to_keep = ['IsHoliday_x', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'CPI', 'Unemployment', 'Weekly_Sales']
    training_df = training_df[cols_to_keep].dropna()


    # Initialize model, loss, and optimizer
    model = NeuralNetwork(
        input_df=training_df[cols_to_keep[:-1]],
        target_df=training_df[cols_to_keep[-1]],
        hidden_layers=(512, 128, 16),
        split_data=0.2,
        learning_rate=0.001
    ).to(device)

    criterion = model.loss_function

    optimizer = model.model_optimizer()

    # Training Loop
    for epoch in range(model.num_epochs):
        model.train_model()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader = model.validation_data(shuffle=True)
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Print statistics
        _update = f'Epoch {epoch + 1}/{model.num_epochs} | Train Loss: {model.train_loss:.4f}'
        if val_loader is not None:
            val_loss = val_loss / len(val_loader.dataset)
            print(f'{_update} | Val Loss: {val_loss:.4f}')
        else:
            val_loss = None
            print(_update)

    # Save model
    torch.save(model.state_dict(), 'model.pth')
