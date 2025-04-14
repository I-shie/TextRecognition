import cv2
import torch
import pytesseract
import supervision as sv
from ultralytics import YOLO

# Optional: Set tesseract path (Windows users only)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 1. Load YOLO model
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
model = YOLO('yolov10s-doclaynet.pt')

# 2. Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 3. Supervision annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(
        processed_frame,
        imgsz=1024,
        conf=0.2,
        iou=0.8
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    # Filter detections for text classes (you can refine this later)
    text_detections = sv.Detections(
        xyxy=detections.xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id
    )

    annotated_frame = frame.copy()
    labels = []

    for i, (xyxy, conf, cls_id) in enumerate(zip(text_detections.xyxy, text_detections.confidence, text_detections.class_id)):
        x1, y1, x2, y2 = map(int, xyxy)
        roi = frame[y1:y2, x1:x2]

        # Preprocess ROI for better OCR (optional)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR on the ROI
        extracted_text = pytesseract.image_to_string(thresh_roi, config='--psm 6')  # psm 6: Assume a block of text

        # Clean up text
        extracted_text = extracted_text.strip().replace("\n", " ")

        # Store label
        label = f"{model.names[cls_id]}: {extracted_text if extracted_text else 'No text'}"
        labels.append(label)

    # Draw boxes
    annotated_frame = box_annotator.annotate(annotated_frame, text_detections)
    
    # Draw labels
    if labels:
        annotated_frame = label_annotator.annotate(annotated_frame, text_detections, labels)

    cv2.imshow("YOLO + OCR", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
