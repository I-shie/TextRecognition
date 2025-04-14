import cv2
import torch
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
import time

# Check GPU availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# 1. Load YOLO model with GPU acceleration
original_load = torch.load
torch.load = lambda args, **kwargs: original_load(*args, **{*kwargs, 'weights_only': False})
model = YOLO('yolov10n-doclaynet.pt').to('cuda')  # Move model to GPU

# 2. Initialize PaddleOCR with GPU support
# Enable GPU with use_gpu=True
# lang: 'en' for English, 'ch' for Chinese, or others as needed
# use_angle_cls=True helps with document orientation
# show_log=False reduces console output
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)

# 3. Initialize webcam with higher FPS settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set higher FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce buffer size for lower latency

# 4. Supervision annotators with optimized parameters
box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.4
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.4,
    text_padding=2
)

# 5. Performance tracking variables
frame_count = 0
start_time = time.time()
prev_labels = []
prev_detections = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Skip YOLO processing every other frame to improve speed
    if frame_count % 2 == 0 or not prev_detections:
        # Convert to RGB and move to GPU
        with torch.no_grad():  # Disable gradient calculation for inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection on GPU
            results = model(rgb_frame, imgsz=1024, conf=0.2, iou=0.8, device='cuda')[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter detections for text regions (adjust class IDs as needed)
            text_detections = sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id
            )
            
            # Process OCR for each detected text region in parallel batches
            labels = []
            rois = []
            coordinates = []
            
            for xyxy, conf, cls_id in zip(text_detections.xyxy, 
                                        text_detections.confidence, 
                                        text_detections.class_id):
                x1, y1, x2, y2 = map(int, xyxy)
                rois.append(frame[y1:y2, x1:x2])
                coordinates.append((x1, y1, x2, y2))
            
            # Batch process ROIs with PaddleOCR
            if rois:
                batch_results = ocr.ocr(rois, cls=True)
                
                for i, ocr_result in enumerate(batch_results):
                    extracted_text = ""
                    if ocr_result and ocr_result[0]:
                        # Join all detected text lines with confidence > 0.5
                        extracted_text = " ".join([line[1][0] for line in ocr_result[0] if line[1][1] > 0.5])
                    
                    # Create label
                    x1, y1, x2, y2 = coordinates[i]
                    label = f"{model.names[text_detections.class_id[i]]}: {extracted_text if extracted_text else 'No text'}"
                    labels.append(label)
            
            prev_labels = labels
            prev_detections = text_detections
    else:
        # Reuse previous frame's detections and labels
        text_detections = prev_detections
        labels = prev_labels

    # Draw annotations
    annotated_frame = box_annotator.annotate(frame.copy(), text_detections)
    if labels:
        annotated_frame = label_annotator.annotate(annotated_frame, text_detections, labels)

    # Calculate and display FPS
    if frame_count % 10 == 0:
        fps = frame_count / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("GPU Accelerated YOLOv10 + PaddleOCR", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Calculate final FPS
end_time = time.time()
total_fps = frame_count / (end_time - start_time)
print(f"Average FPS: {total_fps:.2f}")

cap.release()
cv2.destroyAllWindows()
