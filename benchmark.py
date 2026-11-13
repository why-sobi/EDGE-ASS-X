import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort
from torch.utils.benchmark import Timer

from pipeline_support import (
    RESULTS_DIR,
    ROOT_DIR,
    SAVE_DIR,
    build_model,
    load_data,
    test_model,
)


def count_nonzero(model):
    return sum(int((parameter != 0).sum()) for parameter in model.parameters())


def count_macs(model, sample):
    macs = 0

    def count_layer(module, _inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += (
                output.shape[-2]
                * output.shape[-1]
                * module.out_channels
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features

    handles = [
        module.register_forward_hook(count_layer)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    with torch.no_grad():
        model.cpu().eval()(sample)
    for handle in handles:
        handle.remove()
    return macs


def benchmark_torchscript(names, x_test, y_test):
    sample = x_test[:1]
    rows = []
    predictions = pd.DataFrame({"label": y_test.numpy()})
    for name in names:
        path = SAVE_DIR / f"{name}.ts"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}. Run the model export stages first."
            )
        model = torch.jit.load(str(path)).eval()
        started = time.perf_counter()
        accuracy, predicted = test_model(model, x_test, y_test, "cpu")
        elapsed = time.perf_counter() - started
        latency = Timer(
            "with torch.no_grad(): model(sample)",
            globals={"model": model, "sample": sample, "torch": torch},
            num_threads=4,
        ).timeit(100).mean
        rows.append(
            {
                "model": name,
                "top1": accuracy,
                "latency_ms": latency * 1000,
                "images_per_s": len(y_test) / elapsed,
                "file_mb": path.stat().st_size / 1e6,
            }
        )
        predictions[name] = predicted.numpy()
    return pd.DataFrame(rows), predictions


def model_for_stats(name):
    if name == "structured":
        return torch.load(
            SAVE_DIR / "structured_full_model.pth",
            map_location="cpu",
            weights_only=False,
        )
    model = build_model()
    checkpoint = (
        SAVE_DIR / "unstructured_weights.pth"
        if name == "unstructured"
        else SAVE_DIR / "fp32_weights.pth"
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


def benchmark_onnx(x_test, y_test):
    rows = []
    for name in ("fp32", "ptq_int8", "qat_int8", "unstructured", "structured"):
        path = SAVE_DIR / f"{name}.onnx"
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        outputs = []
        for batch in np.array_split(x_test.numpy(), 10):
            outputs.append(
                session.run(None, {"x": batch.astype(np.float32)})[0]
            )
        logits = np.concatenate(outputs)
        rows.append(
            {
                "model": name,
                "onnx_top1": round(
                    float((logits.argmax(1) == y_test.numpy()).mean()) * 100,
                    2,
                ),
                "onnx_mb": path.stat().st_size / 1e6,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "onnx_results.csv", index=False)


def main():
    torch.set_num_threads(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    _, _, x_test, y_test, _ = load_data()

    q1, q1_predictions = benchmark_torchscript(
        ["fp32", "ptq_int8", "qat_int8"], x_test, y_test
    )
    q1.to_csv(RESULTS_DIR / "q1_benchmark.csv", index=False)
    q1_predictions.to_csv(RESULTS_DIR / "q1_predictions.csv")

    q2, q2_predictions = benchmark_torchscript(
        ["fp32", "unstructured", "structured"], x_test, y_test
    )
    q2_stats = []
    sample = x_test[:1]
    for name in ("fp32", "unstructured", "structured"):
        model = model_for_stats(name)
        q2_stats.append(
            {
                "nonzero_params": count_nonzero(model),
                "macs": count_macs(model, sample),
            }
        )
    q2 = pd.concat(
        [q2.reset_index(drop=True), pd.DataFrame(q2_stats)], axis=1
    )
    q2.to_csv(RESULTS_DIR / "q2_benchmark.csv", index=False)
    q2_predictions.to_csv(RESULTS_DIR / "q2_predictions.csv")

    final = pd.concat(
        [q1, q2[q2.model != "fp32"]], ignore_index=True
    )
    final["reward"] = final.top1 * ((final.file_mb / 12) ** -0.07)
    final.to_csv(RESULTS_DIR / "all_models.csv", index=False)
    benchmark_onnx(x_test, y_test)
    print(q1.to_string(index=False))
    print(q2.to_string(index=False))
    print(final.sort_values("reward", ascending=False).to_string(index=False))
    print(f"Results saved under {ROOT_DIR / 'results'}")


if __name__ == "__main__":
    main()
