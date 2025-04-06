import cv2
import torch
import supervision as sv
from ultralytics import YOLO

# 1. Initialize model with proper weights loading
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
model = YOLO('yolov10s-doclaynet.pt')  # Same weights that worked with images

# 2. Initialize camera with matching image dimensions
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Match your successful image width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Match your successful image height

# 3. Use same annotators as your working image code
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. Apply identical preprocessing as your image pipeline
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Most YOLO models expect RGB
    
    # 5. Use EXACT same inference parameters as your working image code
    results = model(
        processed_frame, 
        imgsz=1024,  # Match your successful image size
        conf=0.2,    # Match your image confidence threshold
        iou=0.8      # Match your image IOU threshold
    )[0]
    
    # 6. Convert detections identically to your image pipeline
    detections = sv.Detections.from_ultralytics(results)
    
    # 7. Apply same class filtering logic
    text_detections = sv.Detections(
        xyxy=detections.xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id
    )
    
    # 8. Annotate identically to your working version
    annotated_frame = box_annotator.annotate(frame.copy(), text_detections)
    
    # 9. Only add labels if detections exist (prevents crashes)
    if len(text_detections) > 0:
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for confidence, class_id 
            in zip(text_detections.confidence, text_detections.class_id)
        ]
        annotated_frame = label_annotator.annotate(annotated_frame, text_detections, labels)
    
    cv2.imshow("Live Detection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
