# usage: python run_inference.py <fp32|ptq_int8|qat_int8|unstructured|structured> [n_images]
import sys, pickle, numpy as np, onnxruntime as ort
from pathlib import Path
name, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000
root = "../data/cifar-10-batches-py" if Path("../data/cifar-10-batches-py").exists() else "data/cifar-10-batches-py"
d = pickle.load(open(f"{root}/test_batch", "rb"), encoding="bytes")
x = d[b"data"][:n].reshape(-1, 3, 32, 32) / 255
x = (x - np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)) / np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
pred = ort.InferenceSession(f"saved_models/{name}.onnx").run(None, {"x": x.astype(np.float32)})[0].argmax(1)
print(f"{name}: accuracy on {n} test images = {(pred == np.array(d[b'labels'][:n])).mean() * 100:.2f}%")
