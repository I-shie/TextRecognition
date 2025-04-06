import cv2
import torch
import supervision as sv
from ultralytics import YOLO

# 1. Initialize model with proper weights loading
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
model = YOLO('yolov10b-doclaynet.pt')  # Make sure this is the correct path

# 2. Initialize camera
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening camera"

# 3. Use updated annotator names (as per Supervision 0.26.0+)
box_annotator = sv.BoxAnnotator()  # Changed from BoundingBoxAnnotator
label_annotator = sv.LabelAnnotator()

# 4. Main processing loop
while True:
    ret, frame = cap.read()
	
    if not ret:
        break
    
    # 5. Run inference
    results = model(frame, conf=0.5, iou=0.5)[0]
    
    # 6. Convert to Supervision Detections properly
    detections = sv.Detections.from_ultralytics(results)
    
    # 7. Filter text detections (adjust class_id as needed)
    text_detections = sv.Detections(
        xyxy=detections.xyxy[detections.class_id == 0],  # Class 0 for text
        confidence=detections.confidence[detections.class_id == 0],
        class_id=detections.class_id[detections.class_id == 0]
    )
    
    # 8. Annotate frame
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=text_detections
    )
    
    # 9. Add labels only if detections exist
    if len(text_detections) > 0:
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for confidence, class_id in zip(text_detections.confidence, text_detections.class_id)
        ]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=text_detections,
            labels=labels
        )
    
    # 10. Display
    cv2.imshow("Live Text Detection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()