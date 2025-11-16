import cv2
from ultralytics import YOLO
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent

# Path to YOLO model
MODEL_PATH = SCRIPT_DIR / "runs" / "detect" / "stage2_finetune_all" / "weights" / "best.pt"

# Confidence threshold:
CONF_THRESHOLD = 0.2

# --- END OF CONFIGURATION ---


def run_webcam_detection():
    """
    Loads the trained YOLO model and runs object detection on the default webcam feed.
    """
    
    # Check if the model file exists
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    
    # Load the trained YOLO model
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open a connection to the default webcam (usually 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting webcam feed. Press 'q' to quit.")
    
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # Read a new frame from the webcam
        success, frame = cap.read()

        if not success:
            print("Error: Failed to capture frame.")
            break

        # # --- FPS Calculation ---
        # new_frame_time = time.time()
        # fps = 1 / (new_frame_time - prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps_text = f"FPS: {int(fps)}"
        # # --- End FPS Calculation ---

        # Run the YOLO model on the frame
        results = model(frame, stream=True, conf=CONF_THRESHOLD, verbose=False)

        for res in results:

            annotated_frame = res.plot()

            # # Put the FPS text on the frame
            # cv2.putText(annotated_frame, fps_text, (10, 30), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Detection (Press 'q' to quit)", annotated_frame)

        # Check if the 'q' key was pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")


if __name__ == "__main__":
    run_webcam_detection()