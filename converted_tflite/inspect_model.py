
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "model_unquant.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n========== INPUT ==========")

for item in input_details:
    print("Name:", item["name"])
    print("Shape:", item["shape"])
    print("Data type:", item["dtype"])
    print("Quantization:", item["quantization"])

print("\n========== OUTPUT ==========")

for item in output_details:
    print("Name:", item["name"])
    print("Shape:", item["shape"])
    print("Data type:", item["dtype"])
    print("Quantization:", item["quantization"])

print("\n========== MODEL ==========")
print("Inputs:", len(input_details))
print("Outputs:", len(output_details))
