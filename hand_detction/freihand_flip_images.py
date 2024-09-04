import cv2
from os import listdir

# Flip training images
num_rgb = len(listdir("../data/freihand/right/training/rgb/"))
num_mask = len(listdir("../data/freihand/right/training/mask/"))

print(num_rgb)
exit()
for idx in range(num_rgb):
    print(f"{idx}/{num_rgb}")
    img_right = cv2.imread(f"../data/freihand/right/training/rgb/{idx:08d}.jpg")
    if idx < num_mask:
        mask_right = cv2.imread(f"../data/freihand/right/training/mask/{idx:08d}.jpg")
    
    img_left = cv2.flip(img_right, 1)
    mask_left = cv2.flip(mask_right, 1)

    cv2.imwrite(f"../data/freihand/left/training/rgb/{idx:08d}.jpg", img_left)
    if idx < num_mask:
        cv2.imwrite(f"../data/freihand/left/training/mask/{idx:08d}.jpg", mask_left)