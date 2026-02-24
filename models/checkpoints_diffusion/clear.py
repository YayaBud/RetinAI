from PIL import Image
import os
from glob import glob

bad_files = []

for folder in ["./train", "./val"]:
    for path in glob(folder + "/*.jpeg"):
        try:
            Image.open(path).verify()
        except:
            bad_files.append(path)

print("Corrupted:", len(bad_files))

for f in bad_files:
    os.remove(f)

print("Removed corrupted files.")
