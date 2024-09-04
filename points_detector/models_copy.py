import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
from datetime import datetime


class TimeRemainingCallback:
    def __init__(self, total_epochs, start_time):
        self.total_epochs = total_epochs
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs=None):
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        average_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = self.total_epochs - (epoch + 1)
        remaining_time = average_epoch_time * remaining_epochs
        print(f"Loss: {logs['loss']:.4f}, Mean Absolute Error: {logs['mae']}, Time remaining: {remaining_time}")


class PointLabellingModel(nn.Module):
    '''
    A machine learning model built upon resnet50
    The model can take in an image of any size and output a series 21 xyz coordinates \
        representing 21 landmarks of a hand
    '''
    def __init__(self):
        super(PointLabellingModel, self).__init__()
        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the classifier
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Additional layers for your specific task
        self.additional_layers = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 21 * 3)  # Each landmark has x, y, and z coordinates
        )

    def forward(self, x):
        # Forward pass through the ResNet backbone
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # Additional layers specific to your task
        x = self.additional_layers(x)

        return x.view(-1, 21, 3)

    def train_model(self, train, test, epochs=10, early_stopping_patience=5, dropout_rate=0.2, learning_rate=0.0001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.to(device)

        criterion = nn.L1Loss()  # Mean Absolute Error
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        start_time = datetime.now()

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        training_info = {'loss': [], 'mae': []}

        for epoch in range(epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            epoch_loss = 0.0
            self.train()

            for inputs, labels in tqdm(train, desc=f"Epoch {epoch + 1}/{epochs} (Training)"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(inputs).float()
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                del inputs, labels, loss

            average_loss = epoch_loss / len(train)

            with torch.no_grad():
                total_mae = 0

                self.eval()
                for inputs, labels in test:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)

                    mae = nn.L1Loss()(outputs, labels)
                    total_mae += mae.item()

            average_mae = total_mae / len(test)

            training_info['loss'].append(average_loss)
            training_info['mae'].append(average_mae)

            time_remaining_callback = TimeRemainingCallback(epochs, start_time)
            time_remaining_callback.on_epoch_end(epoch, {'loss': average_loss, 'mae': average_mae})

            if average_loss < best_loss and average_loss > 0.0001:
                best_loss = average_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                    print(f"Best Validation Loss: {best_loss:.4f} at epoch {best_epoch}")
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                    break

            scheduler.step(average_loss)

        return training_info
