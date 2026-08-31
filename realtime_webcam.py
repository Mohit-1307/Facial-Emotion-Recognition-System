#!/usr/bin/env python3
"""
Real-time facial emotion recognition from a live webcam feed (the project
brief's "Real-Time Processing" requirement).

Pipeline per frame:
    1. Grab a frame from the camera (or --source video/image for testing).
    2. Detect faces with OpenCV's bundled Haar-cascade frontal-face detector
        (fast, dependency-free, well-suited to CPU real-time use -- a DNN
        face detector would be more accurate but is overkill for this stage
        and adds another model to download/optimize).
    3. Crop + preprocess each face to match the trained model's expected
        input (48x48 grayscale for --model scratch, 160x160 RGB for
        --model transfer -- see config.TRANSFER_INPUT_SIZE, the single
        source of truth this script always reads from).
    4. Run the classifier, overlay the predicted emotion + confidence and
        an emoji on the frame.
    5. Show a rolling FPS counter (ties into the "Performance Optimization"
        requirement -- this is exactly the number that matters for
        real-time capability).

Usage
-----
    python realtime_webcam.py --model scratch
    python realtime_webcam.py --model transfer --checkpoint saved_models/transfer_finetune_best.keras
    python realtime_webcam.py --model scratch --source path/to/video.mp4      # offline testing
    python realtime_webcam.py --model scratch --source path/to/photo.jpg     # single-image smoke test

Press 'q' to quit the live window.

Note on this sandbox: the environment this script was developed in has no
camera device attached, so the live-webcam path (--source unset, the
default) cannot be exercised here. The --source flag exists specifically so
the detection + classification + overlay pipeline can still be verified
end-to-end against a static image/video, which is what
tests/test_pipeline.py does. On your own machine with a webcam, just omit --source.
"""

import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from deepfer import config

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class EmotionClassifier:

    def __init__(self, checkpoint_path: str, kind: str):

        self.kind = kind

        self.model = tf.keras.models.load_model(checkpoint_path)

        if kind == "scratch":

            self.input_size = config.SCRATCH_INPUT_SIZE

        else:

            self.input_size = config.TRANSFER_INPUT_SIZE

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:

        if self.kind == "scratch":

            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)

            arr = resized.astype("float32") / 255.0

            arr = arr[..., np.newaxis]  # (H, W, 1)

        else:

            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype("float32")

            arr = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)

        return arr[np.newaxis, ...]  # add batch dim

    def predict(self, face_bgr: np.ndarray):

        x = self.preprocess(face_bgr)

        probs = self.model.predict(x, verbose=0)[0]

        idx = int(np.argmax(probs))

        return config.CLASS_NAMES[idx], float(probs[idx]), probs


def draw_overlay(frame, box, label, confidence):

    x, y, w, h = box

    color = (60, 200, 60)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = f"{label} {confidence * 100:.0f}%"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)

    cv2.putText(
        frame, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )


MIN_DETECTION_SIDE = 300


def detect_faces(frame, face_cascade):
    """
    Runs the Haar-cascade face detector on `frame` and returns boxes in
    the ORIGINAL frame's coordinate system.

    Two fixes over a naive detectMultiScale call, both needed for small or
    tightly-cropped inputs (the bundled sample_images are 48x48 -- exactly
    the case this was breaking on):
    1. Upscale the frame first if it's smaller than MIN_DETECTION_SIDE on
        its short side. A face that fills an entire tiny frame has no
        margin for the cascade's sliding window to work with; upscaling
        gives it room without changing what's actually in the image.
    2. Histogram-equalize the grayscale image for contrast, and derive
        minSize from the *working* resolution rather than a hardcoded
        48px floor, which silently rejected any face smaller than that in
        the original frame.
    """

    h0, w0 = frame.shape[:2]

    short_side = min(h0, w0)

    scale = max(1.0, MIN_DETECTION_SIDE / short_side)

    if scale > 1.0:

        working = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

    else:

        working = frame

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    min_side = max(20, int(0.15 * min(working.shape[:2])))

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
    )

    # Map detections back to the original frame's coordinates.
    return [
        (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        for (x, y, w, h) in faces
    ]


def process_frame(frame, face_cascade, classifier: EmotionClassifier):

    faces = detect_faces(frame, face_cascade)

    results = []

    for x, y, w, h in faces:

        face = frame[y : y + h, x : x + w]

        label, conf, probs = classifier.predict(face)

        draw_overlay(frame, (x, y, w, h), label, conf)

        results.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "label": label,
                "confidence": conf,
            }
        )

    return frame, results


def run_on_image(path, classifier, out_path=None):

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    frame = cv2.imread(path)

    if frame is None:

        raise SystemExit(f"Could not read image: {path}")

    frame, results = process_frame(frame, face_cascade, classifier)

    out_path = out_path or "realtime_demo_output.jpg"

    cv2.imwrite(out_path, frame)

    print(f"Detections: {results}")

    print(f"Annotated image written to {out_path}")

    return results


def run_on_video_or_camera(source, classifier, max_frames=None):

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(source if source is not None else 0)

    if not cap.isOpened():

        raise SystemExit(
            f"Could not open video source: {source if source is not None else '(default camera)'}"
        )

    frame_count = 0

    t_start = time.time()

    fps = 0.0

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        frame, _ = process_frame(frame, face_cascade, classifier)

        frame_count += 1

        elapsed = time.time() - t_start

        if elapsed > 0.5:

            fps = frame_count / elapsed

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow("DeepFER - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

        if max_frames and frame_count >= max_frames:

            break

    cap.release()

    cv2.destroyAllWindows()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--model", choices=["scratch", "transfer"], default="scratch")

    ap.add_argument(
        "--checkpoint", default=None, help="Defaults to saved_models/<model>_best.keras"
    )

    ap.add_argument(
        "--source",
        default=None,
        help="Path to an image or video file for offline testing. Omit to use the live webcam.",
    )

    ap.add_argument(
        "--out", default=None, help="Output path when --source is a still image"
    )

    args = ap.parse_args()

    checkpoint = args.checkpoint or str(
        config.SAVED_MODELS_DIR / f"{args.model}_best.keras"
    )

    classifier = EmotionClassifier(checkpoint, args.model)

    if args.source and args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

        run_on_image(args.source, classifier, args.out)

    else:

        run_on_video_or_camera(args.source, classifier)


if __name__ == "__main__":

    main()
