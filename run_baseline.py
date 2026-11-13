import argparse
import random

import torch

from pipeline_support import (
    RESULTS_DIR,
    SAVE_DIR,
    build_model,
    fit_model,
    load_data,
    test_model,
)


def main():
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 FP32 baseline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train again even if the baseline checkpoint already exists.",
    )
    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)
    SAVE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test, _ = load_data()
    model = build_model()
    checkpoint = SAVE_DIR / "fp32_weights.pth"
    if checkpoint.is_file() and not args.force:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"Loaded existing baseline checkpoint: {checkpoint}")
    else:
        fit_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=25,
            learning_rate=0.1,
        )
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved baseline checkpoint: {checkpoint}")

    accuracy, _ = test_model(model, x_test, y_test, "cpu")
    print(f"FP32 accuracy: {accuracy}%")
    sample = x_test[:1]
    model = model.cpu().eval()
    torch.jit.save(
        torch.jit.trace(model, sample),
        SAVE_DIR / "fp32.ts",
    )
    torch.onnx.export(
        model,
        sample,
        SAVE_DIR / "fp32.onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    print("Saved baseline TorchScript and ONNX exports.")


if __name__ == "__main__":
    main()
