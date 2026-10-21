import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"
IMAGE_PATH = "test.jpg"


# -----------------------------
# Load labels
# -----------------------------
with open(LABELS_PATH, "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]

print("Labels:")
for i, label in enumerate(labels):
    print(f"  {i}: {label}")


# -----------------------------
# Load model
# -----------------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

height = input_shape[1]
width = input_shape[2]

print(f"\nModel input: {width}x{height}")


# -----------------------------
# Load and prepare image
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

print(f"Original image size: {image.size}")

image = image.resize((width, height))

image_array = np.asarray(image, dtype=np.float32)

# Teachable Machine float models expect
# pixel values normalized to approximately -1 to 1.
image_array = (image_array / 127.5) - 1.0

# Add batch dimension
input_data = np.expand_dims(image_array, axis=0)

print("Input tensor shape:", input_data.shape)


# -----------------------------
# Run inference
# -----------------------------
interpreter.set_tensor(
    input_details[0]["index"],
    input_data
)

interpreter.invoke()


# -----------------------------
# Get predictions
# -----------------------------
output_data = interpreter.get_tensor(
    output_details[0]["index"]
)[0]

print("\n========== PREDICTIONS ==========")

for i, score in enumerate(output_data):
    print(f"{labels[i]:30} {score:.4f} ({score * 100:.2f}%)")


# -----------------------------
# Best prediction
# -----------------------------
best_index = np.argmax(output_data)

best_label = labels[best_index]
best_score = output_data[best_index]

print("\n========== RESULT ==========")
print(f"Prediction : {best_label}")
print(f"Confidence : {best_score * 100:.2f}%")
