import io
import time
import urllib.request

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


# ============================================================
# CONFIGURATION
# ============================================================

ESP32_CAPTURE_URL = "http://192.168.0.111/capture"

MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r") as f:
    labels = [
        line.strip().split(" ", 1)[1]
        for line in f
        if line.strip()
    ]

print("Labels:")

for i, label in enumerate(labels):
    print(f"  {i}: {label}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading AI model...")

interpreter = Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

height = input_shape[1]
width = input_shape[2]

print(f"Model input: {width}x{height}")


# ============================================================
# CAPTURE IMAGE FROM ESP32-CAM
# ============================================================

def capture_from_esp32():

    print("\nCapturing image from ESP32-CAM...")
    print(f"URL: {ESP32_CAPTURE_URL}")

    try:

        request = urllib.request.Request(
            ESP32_CAPTURE_URL,
            headers={
                "Accept": "image/jpeg"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            image_bytes = response.read()

        print(
            f"Received {len(image_bytes)} bytes"
        )

        if not image_bytes:
            raise RuntimeError(
                "ESP32 returned an empty image"
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print(
            f"Camera image size: {image.size}"
        )

        return image

    except Exception as e:

        print(
            f"ERROR: Could not capture image: {e}"
        )

        return None


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    image = image.resize(
        (width, height)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Teachable Machine float model:
    # normalize pixels from 0-255 to approximately -1 to +1
    image_array = (
        image_array / 127.5
    ) - 1.0

    # Add batch dimension
    input_data = np.expand_dims(
        image_array,
        axis=0
    )

    return input_data


# ============================================================
# RUN AI INFERENCE
# ============================================================

def predict(image):

    input_data = prepare_image(image)

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    output_data = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    return output_data


# ============================================================
# MAIN
# ============================================================

print("\n========================================")
print(" ESP32-CAM → RASPBERRY PI → AI")
print("========================================")

start_time = time.time()

image = capture_from_esp32()

if image is None:
    print("\nPrediction aborted.")
    exit(1)


# Run inference
output_data = predict(image)

inference_time = time.time() - start_time


# ============================================================
# DISPLAY ALL PREDICTIONS
# ============================================================

print("\n========== PREDICTIONS ==========")

for i, score in enumerate(output_data):

    print(
        f"{labels[i]:30} "
        f"{score:.4f} "
        f"({score * 100:.2f}%)"
    )


# ============================================================
# BEST PREDICTION
# ============================================================

best_index = np.argmax(
    output_data
)

best_label = labels[best_index]

best_score = output_data[
    best_index
]


print("\n========== RESULT ==========")

print(
    f"Prediction : {best_label}"
)

print(
    f"Confidence : {best_score * 100:.2f}%"
)

print(
    f"Processing : {inference_time:.3f} seconds"
)

print(
    "========================================"
)
