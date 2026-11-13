import copy
import random

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from pipeline_support import (
    RESULTS_DIR,
    SAVE_DIR,
    load_baseline,
    load_data,
    test_model,
)


def main():
    torch.backends.quantized.engine = "fbgemm"
    torch.manual_seed(1)
    random.seed(1)
    SAVE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test, _ = load_data()
    baseline = load_baseline()

    def quantize(calibration_size):
        model = copy.deepcopy(baseline).cpu().eval()
        model.fuse_model()
        model.qconfig = get_default_qconfig("fbgemm")
        model = prepare(model)
        indices = torch.randperm(len(x_train))[:calibration_size]
        with torch.no_grad():
            for batch in x_train[indices].split(100):
                model(batch)
        return convert(model)

    calibrated = {}
    for size in (100, 1000, 5000):
        model = quantize(size)
        accuracy, _ = test_model(model, x_test, y_test)
        print(f"PTQ with {size} calibration images: accuracy {accuracy}%")
        calibrated[size] = model

    model = calibrated[1000]
    torch.save(model.state_dict(), SAVE_DIR / "ptq_int8_weights.pth")
    sample = x_test[:1]
    torch.jit.save(
        torch.jit.trace(model.eval(), sample),
        SAVE_DIR / "ptq_int8.ts",
    )
    torch.onnx.export(
        model.eval(),
        sample,
        SAVE_DIR / "ptq_int8.onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    print("Saved PTQ checkpoint, TorchScript, and ONNX exports.")


if __name__ == "__main__":
    main()
