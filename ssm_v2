import cv2
import mediapipe as mp
import pytesseract
import argostranslate.package, argostranslate.translate
from gtts import gTTS
from datetime import datetime
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ——— Configuration ———
mp_hands        = mp.solutions.hands
mp_drawing      = mp.solutions.drawing_utils
PINCH_THRESH    = 0.05    # normalized pinch threshold
DEBOUNCE_FRAMES = 5       # how many frames needed to confirm a pinch
SMOOTH_WINDOW   = 5       # how many samples to average for corner pos

# ——— Offline translation setup (Argos Translate) ———
# ensure you have downloaded 'en_hi.argosmodel' into your working dir
argostranslate.package.install_from_path("en_hi.argosmodel")
langs       = argostranslate.translate.get_installed_languages()
source_lang = next(l for l in langs if l.code == "en")
target_lang = next(l for l in langs if l.code == "hi")

cap = cv2.VideoCapture(0)

# State
first_corner        = None
rectangles          = []  # (pt1, pt2, translated_text)
pinch_history       = deque(maxlen=DEBOUNCE_FRAMES)
pos_history         = deque(maxlen=SMOOTH_WINDOW)
pinch_confirmed     = False

def get_smoothed_pos():
    if not pos_history:
        return None
    xs = [p[0] for p in pos_history]
    ys = [p[1] for p in pos_history]
    return (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))

FONT = ImageFont.truetype("NotoSansDevanagari-VariableFont_wdth,wght.ttf",20)


def extract_and_ocr(frame, p1, p2):
    # 1) Crop & normalize
    x1, y1 = p1; x2, y2 = p2
    x0, x1 = sorted([x1, x2])
    y0, y1 = sorted([y1, y2])
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        print("⚠️ Empty ROI!")
        return ""

    # 2) Pre‑process for OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8
    )

    # 3) (Optional) Show the pre‑processed ROI for debugging
    cv2.imshow("ROI for OCR", gray)

    # 4) OCR with explicit config
    custom_cfg = r'--oem 1 --psm 6'
    eng_text = pytesseract.image_to_string(gray, config=custom_cfg).strip()
    if not eng_text:
        print("⚠️ Tesseract returned no text.")
        return ""

    print(f"[OCR] {eng_text!r}")

    # 5) Translate offline to Hindi
    translation_obj = source_lang.get_translation(target_lang)
    if translation_obj:
        hi_text = translation_obj.translate(eng_text)
    else:
        print("⚠️ No offline Hindi model; using English.")
        hi_text = eng_text

    print(f"[TRANSLATED] {hi_text!r}")

    # 6) Save the (original) ROI image
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_fname = f"rect_{ts}.png"
    cv2.imwrite(img_fname, roi)
    print(f"[Saved image] {img_fname}")

    # 7) Generate Hindi speech with gTTS
    tts = gTTS(text=hi_text, lang="hi")
    audio_fname = f"rect_{ts}.mp3"
    tts.save(audio_fname)
    print(f"[Saved audio] {audio_fname}")

    return hi_text


with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # you can uncomment to flip if needed
        # frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 1) Detect hand and compute pinch distance + midpoint
        results   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pinch_now = False
        raw_pos   = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            dx, dy = lm[4].x - lm[8].x, lm[4].y - lm[8].y
            if (dx*dx + dy*dy)**0.5 < PINCH_THRESH:
                pinch_now = True
                mx = int((lm[4].x + lm[8].x)/2 * w)
                my = int((lm[4].y + lm[8].y)/2 * h)
                raw_pos = (mx, my)
                pos_history.append(raw_pos)
            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

        # 2) Debounce pinch
        pinch_history.append(pinch_now)
        if len(pinch_history) == DEBOUNCE_FRAMES and all(pinch_history):
            if not pinch_confirmed:
                pinch_confirmed = True
                corner_pos = get_smoothed_pos()
                if corner_pos:
                    if first_corner is None:
                        first_corner = corner_pos
                        print("First corner:", first_corner)
                    else:
                        second_corner = corner_pos
                        translated = extract_and_ocr(frame, first_corner, second_corner)
                        rectangles.append((first_corner, second_corner, translated))
                        first_corner = None
        else:
            if pinch_confirmed and not pinch_now:
                pinch_confirmed = False
                pos_history.clear()

        # 3) Visualize
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for (p1, p2, txt) in rectangles:
            x0, y0 = p1
            # draw all lines of translated text
            for i, line in enumerate(txt.splitlines()):
                draw.text(
                    (x0+5, y0+5 + i*(FONT.size+4)),
                    line,
                    font=FONT,
                    fill=(0,255,0)
                )
        # convert back to OpenCV BGR
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # show active first corner
        if first_corner:
            cv2.circle(frame, first_corner, 6, (0,0,255), -1)

        cv2.imshow("Accurate Pinch‑to‑Rect with Hindi TTS", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
