import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
from datetime import datetime
from losses import Loss_Func


class TimeRemainingCallback:
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


class OccModel(nn.Module):
    """
    A machine learning model built upon resnet50
    The model can take in an image of any size and output a series 21 xyz coordinates \
        representing 21 landmarks of a hand
    """

    def __init__(self):
        super(OccModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.resnet_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[:-1],
            resnet.layer4[:4],
            resnet.avgpool,
        )

        # Additional convolutional layers
        self.conv4e = nn.Sequential(
            nn.Conv2d(resnet.fc.in_features, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv4f = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_stub = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bilinear_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # Used to allow for images of any size
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.additional_layers = nn.Sequential(
            nn.Linear(256, 21 * 3)  # Each landmark has x, y, and z coordinates
        )

        # Initialize additional convolutional layers with random weights
        nn.init.kaiming_normal_(
            self.conv4e[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv4e[1].weight, 1)
        nn.init.constant_(self.conv4e[1].bias, 0)

        nn.init.kaiming_normal_(
            self.conv4f[0].weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv4f[1].weight, 1)
        nn.init.constant_(self.conv4f[1].bias, 0)

        nn.init.kaiming_normal_(
            self.conv_stub.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv_stub.bias, 0)

    def forward(self, x):
        x = self.resnet_layers(x)
        x = self.conv4e(x)
        x = self.conv4f(x)
        x = self.conv_stub(x)
        x = self.bilinear_upsample(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.additional_layers(x)
        return x.view(-1, 21, 3)

    def train_model(
        self,
        train,
        test,
        epochs=10,
        early_stopping_patience=5,
        learning_rate=0.0001,
        loss_type=None,
        loss_weights=[1, 1],
    ):
        """
        Trains the model.

        Args:
            `train`: DataLoader for training data.
            `test`: DataLoader for testing data.
            `epochs`: Number of epochs for training.
            `early_stopping_patience`: Patience for early stopping.
            `learning_rate`: Learning rate used whilst training
            `loss_type`: Type of loss to use
            `loss_weights`: Weighting to apply on losses

        Returns:
            Training information containing loss, and error.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.to(device)

        criterion = Loss_Func(loss_type, loss_weights)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

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

            scheduler.step(average_loss)

        return training_info
