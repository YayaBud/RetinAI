import glob
import shutil
import os
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "healthy_processed")
TRAIN = os.path.join(BASE_DIR, "train")
VAL = os.path.join(BASE_DIR, "val")

os.makedirs(TRAIN, exist_ok=True)
os.makedirs(VAL, exist_ok=True)

images = glob.glob(os.path.join(SRC, "*.jpeg"))

print("Total images found:", len(images))

train_imgs, val_imgs = train_test_split(
    images,
    test_size=0.1,
    random_state=42
)

print("Moving train images...")
for img in train_imgs:
    shutil.move(img, os.path.join(TRAIN, os.path.basename(img)))

print("Moving val images...")
for img in val_imgs:
    shutil.move(img, os.path.join(VAL, os.path.basename(img)))

print("Train:", len(train_imgs))
print("Val:", len(val_imgs))

# Remove source folder if empty
if len(os.listdir(SRC)) == 0:
    os.rmdir(SRC)
