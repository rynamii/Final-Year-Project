import torch
import torch.nn as nn
import torch.optim as optim

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import Loss_Func
from tqdm import tqdm
from datetime import datetime


class TimeRemainingCallback:
    """Callback to display estimated remaining time after each epoch during training."""

    def __init__(self, total_epochs, start_time):
        """
        Initializes TimeRemainingCallback.

        Args:
            `total_epochs`: Total number of epochs for training.
            `start_time`: Start time of training.
        """
        self.total_epochs = total_epochs
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs):
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
            f"Loss: {logs['loss']:.6f}, Mean Absolute Error: {logs['mae']}, Time remaining: {remaining_time}"
        )


class kp2pose(nn.Module):
    """
    Model that converts 21*3 keypoints to 16*3 mano pose (angles)
    """

    def __init__(self):
        super(kp2pose, self).__init__()
        self.bn_input = nn.BatchNorm1d(21 * 3)
        self.fc1 = nn.Linear(21 * 3, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 16 * 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.bn_input(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x.view(-1, 16, 3)

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
            Training information containing loss, and error.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.to(device)

        criterion = Loss_Func()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        start_time = datetime.now()

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        training_info = {"loss": [], "mae": []}

        for epoch in range(epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            epoch_loss = 0.0
            self.train()

            for inputs, labels in tqdm(
                train, desc=f"Epoch {epoch + 1}/{epochs} (Training)"
            ):
                inputs, labels = inputs.to(device).float(), labels.to(device).float()

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels).float()

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

            training_info["loss"].append(average_loss)
            training_info["mae"].append(average_mae)

            time_remaining_callback = TimeRemainingCallback(epochs, start_time)
            time_remaining_callback.on_epoch_end(
                epoch, {"loss": average_loss, "mae": average_mae}
            )

            if average_loss < best_loss:
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

            # scheduler.step(average_loss)

        return training_info
