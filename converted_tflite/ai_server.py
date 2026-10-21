import io
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn


# ============================================================
# CONFIGURATION
# ============================================================

ESP32_CAPTURE_URL = "http://192.168.0.111/capture"

MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"

# How often the ESP32-CAM is analyzed
CAPTURE_INTERVAL = 2.0

# Predictions below this value are considered uncertain
CONFIDENCE_THRESHOLD = 0.70

# Save captured images
SAVE_IMAGES = True
CAPTURE_DIR = Path("captures")

# API
HOST = "0.0.0.0"
PORT = 8000


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Maize AI Detection API",
    version="1.1.0",
    description="ESP32-CAM maize disease detection service",
)


# Allow the dashboard (served from a different port) and mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r") as f:
    labels = [
        line.strip().split(" ", 1)[1]
        for line in f
        if line.strip()
    ]


# ============================================================
# LOAD AI MODEL
# ============================================================

print("Loading AI model...")

interpreter = Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

INPUT_HEIGHT = input_shape[1]
INPUT_WIDTH = input_shape[2]

print(
    f"Model input: "
    f"{INPUT_WIDTH}x{INPUT_HEIGHT}"
)

print("Labels:")

for i, label in enumerate(labels):
    print(f"  {i}: {label}")


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.Lock()

latest_result = {
    "status": "starting",
    "camera": {
        "url": ESP32_CAPTURE_URL,
        "connected": False,
    },
    "detection": None,
    "image": None,
    "timestamp": None,
    "processing_time_seconds": None,
    "error": None,
}

# JPEG bytes of the last successfully analyzed frame (served to the dashboard)
latest_image_bytes = None


# ============================================================
# RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {
    "Maize_Healthy": {
        "title": "Healthy maize leaf",
        "message": (
            "No obvious disease detected. "
            "Continue normal crop monitoring."
        ),
        "severity": "none",
        "action": "Continue monitoring",
    },

    "Maize_Common_Rust": {
        "title": "Common rust detected",
        "message": (
            "Rust-like symptoms detected. "
            "Inspect surrounding plants and monitor "
            "the spread of infection."
        ),
        "severity": "moderate",
        "action": "Inspect plants and monitor spread",
    },

    "Maize_Northern_Leaf_Blight": {
        "title": "Northern leaf blight detected",
        "message": (
            "Symptoms consistent with northern leaf blight "
            "were detected. Inspect affected plants and "
            "consider appropriate crop-management measures."
        ),
        "severity": "high",
        "action": "Inspect affected plants and manage disease",
    },

    "Maize_Gray_Leaf_Spot": {
        "title": "Gray leaf spot detected",
        "message": (
            "Symptoms consistent with gray leaf spot "
            "were detected. Inspect surrounding plants "
            "and monitor disease progression."
        ),
        "severity": "moderate",
        "action": "Inspect plants and monitor spread",
    },
}


# ============================================================
# CAPTURE IMAGE
# ============================================================

def capture_image():

    request = urllib.request.Request(
        ESP32_CAPTURE_URL,
        headers={
            "Accept": "image/jpeg",
            "Cache-Control": "no-cache",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=10,
    ) as response:

        image_bytes = response.read()

    if not image_bytes:
        raise RuntimeError(
            "ESP32 returned an empty image"
        )

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return image, image_bytes


# ============================================================
# PREPROCESS
# ============================================================

def prepare_image(image):

    image = image.resize(
        (
            INPUT_WIDTH,
            INPUT_HEIGHT,
        )
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Teachable Machine float model
    image_array = (
        image_array / 127.5
    ) - 1.0

    return np.expand_dims(
        image_array,
        axis=0,
    )


# ============================================================
# AI INFERENCE
# ============================================================

def run_inference(image):

    input_data = prepare_image(image)

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data,
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    return output_data


# ============================================================
# CREATE RESULT
# ============================================================

def create_detection_result(output_data):

    best_index = int(
        np.argmax(output_data)
    )

    best_score = float(
        output_data[best_index]
    )

    best_label = labels[best_index]

    confidence_percent = round(
        best_score * 100,
        2,
    )

    # Determine confidence state
    if best_score >= CONFIDENCE_THRESHOLD:
        confidence_status = "high"
        detection_status = "detected"

        recommendation = RECOMMENDATIONS.get(
            best_label,
            {
                "title": best_label,
                "message": "Inspect the plant manually.",
                "severity": "unknown",
                "action": "Inspect plant",
            },
        )

    else:
        confidence_status = "low"
        detection_status = "uncertain"

        recommendation = {
            "title": "Uncertain result",
            "message": (
                "The AI is not sufficiently confident "
                "in this prediction. Capture another "
                "image with the leaf clearly visible."
            ),
            "severity": "unknown",
            "action": "Capture another image",
        }

    # All class probabilities
    probabilities = {}

    for i, score in enumerate(output_data):
        probabilities[labels[i]] = round(
            float(score),
            6,
        )

    return {
        "status": detection_status,
        "label": best_label,
        "confidence": best_score,
        "confidence_percent": confidence_percent,
        "confidence_status": confidence_status,
        "probabilities": probabilities,
        "recommendation": recommendation,
    }


# ============================================================
# SAVE IMAGE
# ============================================================

def save_capture(image_bytes):

    if not SAVE_IMAGES:
        return None

    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    )

    filename = timestamp.strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    filename += ".jpg"

    path = CAPTURE_DIR / filename

    path.write_bytes(
        image_bytes
    )

    return str(path)


# ============================================================
# AI DETECTION LOOP
# ============================================================

def detection_loop():

    global latest_result, latest_image_bytes

    print("\n========================================")
    print(" ESP32-CAM AI DETECTION LOOP")
    print("========================================")
    print(
        f"Camera: {ESP32_CAPTURE_URL}"
    )
    print(
        f"Interval: {CAPTURE_INTERVAL}s"
    )
    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )
    print("========================================\n")

    while True:

        cycle_start = time.time()

        try:

            # ----------------------------------------
            # Capture
            # ----------------------------------------

            print(
                "\nCapturing image..."
            )

            capture_start = time.time()

            image, image_bytes = (
                capture_image()
            )

            capture_time = (
                time.time()
                - capture_start
            )

            print(
                f"Camera image: "
                f"{image.size}"
            )

            print(
                f"Image size: "
                f"{len(image_bytes)} bytes"
            )

            # ----------------------------------------
            # Save
            # ----------------------------------------

            image_path = save_capture(
                image_bytes
            )

            # ----------------------------------------
            # AI
            # ----------------------------------------

            inference_start = time.time()

            output_data = run_inference(
                image
            )

            inference_time = (
                time.time()
                - inference_start
            )

            # ----------------------------------------
            # Result
            # ----------------------------------------

            detection = (
                create_detection_result(
                    output_data
                )
            )

            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

            result = {
                "status": "online",

                "camera": {
                    "url": ESP32_CAPTURE_URL,
                    "connected": True,
                },

                "detection": detection,

                "image": {
                    "width": image.width,
                    "height": image.height,
                    "size_bytes": len(image_bytes),
                    "path": image_path,
                    "url": "/api/detection/image",
                },

                "timestamp": timestamp,

                "processing_time_seconds": round(
                    inference_time,
                    4,
                ),

                "capture_time_seconds": round(
                    capture_time,
                    4,
                ),

                "error": None,
            }

            # ----------------------------------------
            # Update global result + image together
            # ----------------------------------------

            with state_lock:
                latest_result = result
                latest_image_bytes = image_bytes

            # ----------------------------------------
            # Terminal output
            # ----------------------------------------

            print(
                "\n========== AI RESULT =========="
            )

            print(
                f"Prediction : "
                f"{detection['label']}"
            )

            print(
                f"Confidence : "
                f"{detection['confidence_percent']:.2f}%"
            )

            print(
                f"Status     : "
                f"{detection['status']}"
            )

            print(
                f"Inference  : "
                f"{inference_time:.3f}s"
            )

            print(
                "===============================\n"
            )

        except Exception as e:

            print(
                f"\nAI detection error: {e}"
            )

            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

            with state_lock:

                latest_result = {
                    "status": "error",

                    "camera": {
                        "url": ESP32_CAPTURE_URL,
                        "connected": False,
                    },

                    "detection": None,

                    "image": None,

                    "timestamp": timestamp,

                    "processing_time_seconds": None,

                    "error": str(e),
                }

        # ----------------------------------------
        # Maintain capture interval
        # ----------------------------------------

        elapsed = (
            time.time()
            - cycle_start
        )

        sleep_time = max(
            0,
            CAPTURE_INTERVAL - elapsed,
        )

        time.sleep(
            sleep_time
        )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Maize AI Detection API",
        "version": "1.1.0",
        "status": "running",
    }


@app.get("/api/health")
def health():

    with state_lock:
        result = dict(latest_result)

    return {
        "api": "online",
        "ai": result["status"],
        "camera_connected": result[
            "camera"
        ]["connected"],
        "timestamp": result[
            "timestamp"
        ],
    }


@app.get("/api/detection/latest")
def latest_detection():

    with state_lock:
        return JSONResponse(
            content=latest_result
        )


@app.get("/api/detection")
def detection():

    with state_lock:
        return JSONResponse(
            content=latest_result
        )


@app.get("/api/detection/image")
def detection_image():
    """Return the JPEG frame that produced the latest successful detection."""

    with state_lock:
        image_bytes = latest_image_bytes

    if image_bytes is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No image analyzed yet"},
        )

    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    # Start AI loop in background
    ai_thread = threading.Thread(
        target=detection_loop,
        daemon=True,
    )

    ai_thread.start()

    print(
        "\nStarting API server..."
    )

    print(
        f"Listening on "
        f"http://{HOST}:{PORT}"
    )

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
    )
