import cv2
from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Set path to input image
TEST_IMAGE_PATH = SCRIPT_DIR / "c4eed5b762216906a42dc23204ebee0f.jpg"  # <-- CHANGE THIS

# Path to YOLO model
MODEL_PATH = SCRIPT_DIR / "training" / "runs" / "detect" / "stage2_finetune_all" / "weights" / "best.pt"

# Output folder for results
SAVE_DIR = SCRIPT_DIR / "test_image_outputs"

# Confidence threshold
CONF_THRESHOLD = 0.5


def run_image_detection():
    """
    Loads the trained YOLO model, runs detection on a specified image,
    displays the results, and saves the annotated image.
    """

    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    if not Path(TEST_IMAGE_PATH).exists():
        print(f"Error: Test image file not found at {TEST_IMAGE_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading image from {TEST_IMAGE_PATH}...")
    try:
        img = cv2.imread(str(TEST_IMAGE_PATH))
        if img is None:
            raise IOError("Image file could not be read. It might be corrupt or an unsupported format.")
        print("Image loaded successfully.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Running detection...")
    results = model(img, conf=CONF_THRESHOLD, verbose=False)

    annotated_frame = None
    if results and results[0]:
        annotated_frame = results[0].plot()
    else:
        print("No results found.")
        annotated_frame = img # Show the original image if nothing is found

    # Ensure the save directory exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    save_filename = f"{Path(TEST_IMAGE_PATH).stem}_annotated.jpg"
    save_path = SAVE_DIR / save_filename
    
    try:
        cv2.imwrite(str(save_path), annotated_frame)
        print(f"\nSuccessfully saved annotated image to:\n{save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

    print("\nDisplaying image. Press any key to quit.")
    try:
        cv2.imshow("YOLOv8 Detection (Press any key to quit)", annotated_frame)
        # Wait indefinitely for a key press
        cv2.waitKey(0)
    except Exception as e:
        print(f"Error displaying image: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Window closed.")


if __name__ == "__main__":
    run_image_detection()