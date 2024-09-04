from torchvision import transforms
from PIL import Image
# import matplotlib.pyplot as plt
import cv2
import numpy

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToPILImage()
        # Add any additional preprocessing steps here
    ]
)

image = Image.open(
    "/home/devcontainers/third-year-project/data/hand_images_synth/full/left/00000001.jpg"
)

augmented_image = train_transform(image)

opencvImage = cv2.cvtColor(numpy.array(augmented_image), cv2.COLOR_RGB2BGR)

cv2.imshow("test", opencvImage)
cv2.waitKey(0)

cv2.imwrite("normalised_image.png", opencvImage)
