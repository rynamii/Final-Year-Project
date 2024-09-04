import cv2
import models
import torch
from torchvision import transforms
import points_displayer
import re
import subprocess

result = subprocess.run(['ipconfig.exe'], capture_output=True, text=True)

# Extract IPv4 address using regex
ipv4_pattern = re.compile(r'IPv4 Address[^\d]+(\d+\.\d+\.\d+\.\d+)')
match = ipv4_pattern.search(result.stdout)
ip = match.group(1)

print("Getting camera")
cap = cv2.VideoCapture(f'http://{ip}:8000')
print("Connected")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.OccModel().to(device)

print("loading")
model.load_state_dict(torch.load("models/points_20240108-013810.pt"))
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor()
])
model.eval()
print("loaded")

cv2.namedWindow('Landmark Detection')

while cap.isOpened():
    flag, image = cap.read()
    new_width = 300

    # Resize the image, letting OpenCV calculate the new height proportionally
    resized_image = cv2.resize(image, (224, 224))

    # min_dim = min(image.shape[0], image.shape[1])
    # start_x = (image.shape[1] - min_dim) // 2
    # start_y = (image.shape[0] - min_dim) // 2
    # cropped_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]

    transformed_image = transform(resized_image) / 255

    joints = model(torch.stack([transformed_image]).to(device))
    print(joints.cpu().detach().numpy().tolist()[0])
    points_displayer.draw_landmarks(image, joints.cpu().detach().numpy().tolist()[0])

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Landmark Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
