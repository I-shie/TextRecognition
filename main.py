import cv2
import torch
import pytesseract
import supervision as sv
from ultralytics import YOLO

# Optional: Set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

# Initialize model with proper weights loading
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
model = YOLO('yolov10s-doclaynet.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Cache last seen boxes to prevent OCR reprocessing
last_boxes = None
last_texts = []

# Function to wrap text into multiple lines
def wrap_text(text, max_line_length=30):
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and len(line + words[0]) <= max_line_length:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())
    return lines

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(rgb_frame, imgsz=1024, conf=0.2, iou=0.8)[0]
    detections = sv.Detections.from_ultralytics(results)

    text_detections = sv.Detections(
        xyxy=detections.xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id
    )

    # Annotate boxes
    annotated_frame = box_annotator.annotate(frame.copy(), text_detections)

    # Check if boxes changed
    current_boxes = text_detections.xyxy if len(text_detections) > 0 else None
    boxes_changed = True
    if last_boxes is not None and current_boxes is not None:
        if last_boxes.shape == current_boxes.shape:
            boxes_changed = not (last_boxes == current_boxes).all()

    # Run OCR only if boxes changed
    if boxes_changed and current_boxes is not None:
        last_texts = []
        for box in current_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray_roi, config='--psm 6')  # PSM 6: Assume a uniform block of text
            last_texts.append(text.strip())
        last_boxes = current_boxes

    # Draw labels inside the bounding boxes (with text wrapping)
    if current_boxes is not None and len(last_texts) == len(current_boxes):
        for i, text in enumerate(last_texts):
            wrapped_lines = wrap_text(text, max_line_length=30)  # Wrap text to fit within box
            x1, y1, x2, y2 = map(int, current_boxes[i])

            # Position for the first line
            y_offset = y1 + 20  # Starting from a small offset from the top of the box
            for line in wrapped_lines:
                cv2.putText(annotated_frame, line, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_offset += 20  # Move down for the next line of text

    cv2.imshow("Live Detection + OCR", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
