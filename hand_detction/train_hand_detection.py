import models
import model_setup
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# Load data using the CustomDataset
loader = model_setup.load_data(max_files=5_000)
torch.save(loader, "preprocessed_data_100.pt")

train_subset, test_subset, val_subset = random_split(loader.dataset,[0.6,0.3,0.1])

val_size = len(val_subset.indices)

train = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test = DataLoader(test_subset,batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val = DataLoader(val_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

hand_detection_model = models.HandDetectionModel(128, 1)
hand_detection_model.train_model(train, test, epochs=10)
hand_detection_model.eval()

class_labels = [0, 1, 2]  # Assuming 0 represents null, 1 represents left hand, and 2 represents right hand

with torch.no_grad():
    correct_predictions = 0
    true_labels = []
    predicted_labels = []
    device = torch.device("cuda")

    # Wrap the loop with tqdm to add a progress bar
    for inputs, labels in tqdm(val, total=len(val), desc="Evaluating"):
        outputs = hand_detection_model(inputs.to(device))
        true_labels.extend(labels.tolist())

        for idx, output in enumerate(outputs):
            predicted_label = torch.argmax(output)
            predicted_labels.append(predicted_label.item())

            # print(f"True Label: {labels[idx]}, Predicted Label: {predicted_label.item()}")

            if predicted_label.item() == labels[idx]:
                correct_predictions += 1

    accuracy = correct_predictions / val_size
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Convert true and predicted labels to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)

# Normalize the confusion matrix to get percentages
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Null", "Left", "Right"], yticklabels=["Null", "Left", "Right"])
plt.title('Confusion Matrix - Percentages')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
