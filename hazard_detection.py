import argparse
import cv2
from ultralytics import YOLO
import time
import threading
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Obstacle Detection on Images, Video, or Live Camera using YOLOv8")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Input source: '0' for webcam, path to video file (mp4, avi), or path to image file."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="Path to the exported YOLOv8 model (ONNX, TorchScript) or .pt"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=320, 
        help="Inference image size"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to save annotated video (mp4) or image (jpg/png)."
    )
    parser.add_argument(
        "--alert-sound",
        action="store_true",
        help="Play a beep when a hazard is detected."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional information."
    )
    return parser.parse_args()

def play_beep():
    # Simple beep for Windows, change for other OS if needed
    for _ in range(2):
        print('\a', end='', flush=True)
        time.sleep(0.2)

def process_frame(model, frame, imgsz, conf, labels):
    try:
        results = model.predict(source=frame, imgsz=imgsz, conf=conf)[0]
        hazard_detected = False
        for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(cls)]
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, f"{label} {score:.2f}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                2
            )
            hazard_detected = True
        return frame, hazard_detected
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, False

def test_camera(camera_index):
    """Test if camera is available and working"""
    print(f"Testing camera index {camera_index}...")
    
    # Try different camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Camera {camera_index} working with backend {backend}")
                    cap.release()
                    return True
                else:
                    print(f"✗ Camera {camera_index} opened but no frame received with backend {backend}")
                    cap.release()
            else:
                print(f"✗ Camera {camera_index} failed to open with backend {backend}")
        except Exception as e:
            print(f"✗ Error testing camera {camera_index} with backend {backend}: {e}")
    
    return False

def main():
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Available model files:")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(('.pt', '.onnx')):
                    print(f"  {os.path.join(root, file)}")
        sys.exit(1)
    
    # Handle source parameter
    if args.source.isdigit():
        source = int(args.source)
        is_camera = True
    else:
        source = args.source
        is_camera = False
    
    # Test camera if using webcam
    if is_camera:
        if not test_camera(source):
            print(f"\nCamera {source} is not working. Trying alternative cameras...")
            for alt_camera in [1, 2, -1]:
                if test_camera(alt_camera):
                    print(f"Using alternative camera {alt_camera}")
                    source = alt_camera
                    break
            else:
                print("No working camera found. Please check your camera connection.")
                sys.exit(1)
    
    print(f"Loading model: {args.model}")
    try:
        model = YOLO(args.model)
        labels = model.names
        print(f"Model loaded successfully. Classes: {list(labels.values())}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Single Image File Support ---
    if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        print(f"Processing image: {source}")
        frame = cv2.imread(source)
        if frame is None:
            raise IOError(f"Cannot read image file {source}")
        annotated, hazard = process_frame(model, frame, args.imgsz, args.conf, labels)
        if hazard:
            cv2.putText(
                annotated, "HAZARD DETECTED!!!!", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                3
            )
            if args.alert_sound:
                threading.Thread(target=play_beep).start()

        scale = 0.25
        display = cv2.resize(annotated, (int(annotated.shape[1]*scale), int(annotated.shape[0]*scale)))
        cv2.imshow("Obstacle Detection", display)
        if args.output:
            cv2.imwrite(args.output, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    # --- End Single Image File Support ---

    # Setup capture
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    # Set camera properties for better performance
    if is_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open source {source}")

    # Get camera info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera info: {width}x{height} @ {fps}fps")

    # Setup writer if needed
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    
    print("Starting detection loop. Press 'q' or 'ESC' to quit, 's' to save screenshot...")
    print("Make sure the 'Obstacle Detection' window is visible on your screen!")
    
    # Create window before the loop
    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection", 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame_count += 1
        
        # Process frame
        annotated, hazard = process_frame(model, frame, args.imgsz, args.conf, labels)
        
        if hazard:
            cv2.putText(
                annotated, "⚠️ HAZARD DETECTED!", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 0, 255), 
                3
            )
            if args.alert_sound:
                threading.Thread(target=play_beep).start()

        # Add debug info
        if args.debug:
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(
                annotated, f"FPS: {current_fps:.1f} | Frame: {frame_count}", 
                (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1
            )

        # Display
        scale = 0.5
        display = cv2.resize(annotated, (int(annotated.shape[1]*scale), int(annotated.shape[0]*scale)))
        cv2.imshow("Obstacle Detection", display)
        
        if writer:
            writer.write(annotated)

        # Handle key presses - use longer wait time for better key detection
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Quit key pressed - stopping detection...")
            break
        elif key == ord('s') or key == ord('S'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"Screenshot saved: {filename}")
        elif key == 27:  # ESC key
            print("ESC key pressed - stopping detection...")
            break

    print(f"Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

