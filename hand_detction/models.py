import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight

# import numpy as np
from tqdm import tqdm
from datetime import datetime


class TimeRemainingCallback:
    """Callback to display estimated remaining time after each epoch during training."""

    def __init__(self, total_epochs, start_time) -> None:
        """
        Initializes TimeRemainingCallback.

        Args:
            `total_epochs`: Total number of epochs for training.
            `start_time`: Start time of training.
        """
        self.total_epochs = total_epochs
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs) -> None:
        """
        Prints estimated remaining time after each epoch.

        Args:
            `epoch`: Current epoch number.
            `logs`: Dictionary containing metrics like loss, validation accuracy, and full accuracy.
        """
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        average_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = self.total_epochs - (epoch + 1)
        remaining_time = average_epoch_time * remaining_epochs
        print(
            f"Loss: {logs['loss']:.4f}, Validation Accuracy: {logs['val_acc']:.4f}, \
                Full Accuracy: {logs['full_acc']}, Time remaining: {remaining_time}"
        )


class HandDetectionModel(nn.Module):
    """Model that detects if hands are present in an image
    Outputs 3 labels: 0(Null), 1(Left), 2(Right)
    A CNN that uses pooling to allow variable input sizes
    """

    def __init__(self):
        super(HandDetectionModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the last classification layer of ResNet
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(resnet.fc.in_features, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 3)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.apply(weights_init)

    def forward(self, x):
        x = self.resnet_features(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train_model(
        self,
        train,
        test,
        epochs=10,
        early_stopping_patience=5,
        learning_rate=0.0001,
    ):
        """
        Trains the model.

        Args:
            `train`: DataLoader for training data.
            `test`: DataLoader for testing data.
            `epochs`: Number of epochs for training.
            `early_stopping_patience`: Patience for early stopping.
            `learning_rate`: Learning rate used whilst training

        Returns:
            Training information containing loss, validation accuracy, and full accuracy.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.to(device)

        # Setup class weights
        labels_array = []
        for _, labels in train:
            labels_array.extend(labels.tolist())

        class_weights = torch.tensor(
            compute_class_weight(
                class_weight="balanced", classes=[0, 1, 2], y=labels_array
            )
        ).float()
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

        start_time = datetime.now()
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        training_info = {"loss": [], "val_acc": [], "full_acc": []}

        for epoch in range(epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            epoch_loss = 0.0
            self.train()

            for inputs, labels in tqdm(
                train, desc=f"Epoch {epoch + 1}/{epochs} (Training)"
            ):
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
                correct_predictions = 0
                null_pred = 0
                left_pred = 0
                right_pred = 0
                total_samples = 0
                total_null = 0
                total_left = 0
                total_right = 0

                self.eval()
                for inputs, labels in test:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_null += (labels == 0).sum().item()
                    total_left += (labels == 1).sum().item()
                    total_right += (labels == 2).sum().item()
                    correct_predictions += (predicted == labels).sum().item()
                    null_pred += ((predicted == labels) & (predicted == 0)).sum().item()
                    left_pred += ((predicted == labels) & (predicted == 1)).sum().item()
                    right_pred += (
                        ((predicted == labels) & (predicted == 2)).sum().item()
                    )

            accuracy = correct_predictions / total_samples
            full_acc = [
                null_pred / total_null,
                left_pred / total_left,
                right_pred / total_right,
            ]

            training_info["loss"].append(average_loss)
            training_info["val_acc"].append(accuracy)
            training_info["full_acc"].append(full_acc)

            time_remaining_callback = TimeRemainingCallback(epochs, start_time)
            time_remaining_callback.on_epoch_end(
                epoch, {"loss": average_loss, "val_acc": accuracy, "full_acc": full_acc}
            )

            if average_loss < best_loss and average_loss > 0.0001:
                best_loss = average_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss."
                    )
                    print(
                        f"Best Validation Loss: {best_loss:.4f} at epoch {best_epoch}"
                    )
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                    break

            scheduler.step(average_loss)

        return training_info
