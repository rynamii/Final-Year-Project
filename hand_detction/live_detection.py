import cv2
import models
import torch
from torchvision import transforms
import torch.nn.functional as F
from collections import deque
import re
import subprocess

result = subprocess.run(['ipconfig.exe'], capture_output=True, text=True)

# Extract IPv4 address using regex
ipv4_pattern = re.compile(r'IPv4 Address[^\d]+(\d+\.\d+\.\d+\.\d+)')
match = ipv4_pattern.search(result.stdout)
ip = match.group(1)

cap = cv2.VideoCapture(f'http://{ip}:8000')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.HandDetectionModel().to(device)

threshold = 0.4
default_window_size = 3
max_window_size = 30

print("loading")
model.load_state_dict(torch.load("models/model_resnet50_with_full.pt"))
transform = transforms.Compose([
    transforms.ToTensor()
])
model.eval()
print("loaded")

classes = ["Null", "Left", "Right"]

# Create a window named 'Hand Detection'
cv2.namedWindow('Hand Detection')

# Create a trackbar to dynamically adjust the threshold
cv2.createTrackbar('Threshold', 'Hand Detection', int(threshold * 100), 100, lambda x: None)

# Create a trackbar to dynamically adjust the window size
cv2.createTrackbar('Window Size', 'Hand Detection', default_window_size, max_window_size, lambda x: None)

# Initialize a deque for each class to store confidence values
confidence_buffers = {class_name: deque(maxlen=default_window_size) for class_name in classes}

while cap.isOpened():
    flag, image = cap.read()
    resized_image = cv2.resize(image.copy(), (224, 224))
    image_tensor = transform(resized_image) / 255
    output = model(torch.stack([image_tensor]).to(device))
    softmax = F.softmax(output, dim=1).tolist()[0]

    # Get the current threshold value from the trackbar
    threshold = cv2.getTrackbarPos('Threshold', 'Hand Detection') / 100.0
    image = cv2.flip(image, 1)

    # Get the current window size value from the trackbar
    window_size = cv2.getTrackbarPos('Window Size', 'Hand Detection')

    # Update confidence buffers for each class
    for class_name, confidence_buffer in confidence_buffers.items():
        confidence_buffer.append(softmax[classes.index(class_name)])
    # Display bars for each class
    bar_height = 20
    for i, (class_name, confidence) in enumerate(zip(classes, softmax)):
        bar_length = int(confidence * 300)  # Scale the length of the bar based on confidence
        color = (0, 255, 0) if i == torch.argmax(output) else (128, 128, 128)
        cv2.putText(image, f"{class_name}:", (10, 50 + i * (bar_height + 5) + bar_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (100, 50 + i * (bar_height + 5)),
                      (100 + bar_length, 50 + i * (bar_height + 5) + bar_height), color, -1)
        cv2.putText(image, f"{confidence:.2f}", (105 + bar_length, 50 + i * (bar_height + 5) + bar_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Compute the average confidence over the last 'window_size' frames
    avg_confidences = {
        class_name: sum(confidence_buffer) / len(confidence_buffer)
        for class_name, confidence_buffer in confidence_buffers.items()
        }

    # Decide the final label based on the threshold
    final_label = (max(avg_confidences, key=avg_confidences.get)
                   if avg_confidences[max(avg_confidences, key=avg_confidences.get)] >= threshold else "Null")
    cv2.putText(image, f"Final Label: {final_label}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
