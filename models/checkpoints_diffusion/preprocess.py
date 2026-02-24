import cv2
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

INPUT_DIR = "./healthy_raw"
OUTPUT_DIR = "./healthy_processed"
IMAGE_SIZE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_single(img_path):
    try:
        img = cv2.imread(img_path)

        if img is None:
            return f"Failed to read: {img_path}"

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to isolate retina
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Crop to largest contour
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            img = img[y:y+h, x:x+w]

        # Resize
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Enhance contrast (Ben Graham preprocessing)
        blur = cv2.GaussianBlur(img, (0, 0), 30)
        img = cv2.addWeighted(img, 4, blur, -4, 128)

        save_path = os.path.join(
            OUTPUT_DIR,
            os.path.basename(img_path)
        )

        success = cv2.imwrite(save_path, img)

        if not success:
            return f"Failed to write: {img_path}"

        return None

    except Exception as e:
        return f"Error processing {img_path}: {e}"


if __name__ == "__main__":

    images = glob(os.path.join(INPUT_DIR, "*.jpeg"))

    print(f"Found {len(images)} images")
    print(f"Using {cpu_count()} CPU cores")

    errors = []

    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(preprocess_single, images), total=len(images)):
            if result is not None:
                errors.append(result)

    if errors:
        print(f"\n⚠ {len(errors)} images had issues.")
        with open("preprocess_errors.txt", "w") as f:
            for err in errors:
                f.write(err + "\n")
    else:
        print("\nPreprocessing complete with no errors.")
