import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from pipeline_support import (
    SAVE_DIR,
    fit_model,
    load_baseline,
    load_data,
    test_model,
)


def nonzero_parameters(model):
    return sum(int((parameter != 0).sum()) for parameter in model.parameters())


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test, _ = load_data()
    original = load_baseline()
    print(
        f"Baseline accuracy: {test_model(original, x_test, y_test)[0]}%; "
        f"non-zero parameters: {nonzero_parameters(original)}"
    )

    unstructured = copy.deepcopy(original)
    targets = [
        module
        for module in unstructured.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    for round_number in range(1, 4):
        for module in targets:
            prune.l1_unstructured(module, "weight", amount=0.37)
        accuracy, _ = test_model(unstructured, x_test, y_test, "cpu")
        print(f"Round {round_number}: accuracy after pruning {accuracy}%")
        fit_model(
            unstructured,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=2,
            learning_rate=0.01,
        )
    for module in targets:
        prune.remove(module, "weight")

    unstructured_accuracy, _ = test_model(unstructured, x_test, y_test, "cpu")
    sparsity = 1 - sum(
        int((module.weight != 0).sum()) for module in targets
    ) / sum(module.weight.numel() for module in targets)
    print(
        f"Unstructured pruning: sparsity {sparsity:.1%}, "
        f"accuracy {unstructured_accuracy}%"
    )

    structured = copy.deepcopy(original).cpu()
    for layer in (
        structured.layer1,
        structured.layer2,
        structured.layer3,
        structured.layer4,
    ):
        for block in layer:
            keep_count = int(block.bn1.num_features * 0.6)
            keep = block.bn1.weight.detach().abs().topk(keep_count).indices
            conv1 = nn.Conv2d(
                block.conv1.in_channels,
                keep_count,
                3,
                block.conv1.stride,
                1,
                bias=False,
            )
            bn1 = nn.BatchNorm2d(keep_count)
            conv2 = nn.Conv2d(
                keep_count,
                block.conv2.out_channels,
                3,
                1,
                1,
                bias=False,
            )
            with torch.no_grad():
                conv1.weight.copy_(block.conv1.weight[keep])
                conv2.weight.copy_(block.conv2.weight[:, keep])
                for attribute in ("weight", "bias", "running_mean", "running_var"):
                    getattr(bn1, attribute).copy_(
                        getattr(block.bn1, attribute)[keep]
                    )
            block.conv1, block.bn1, block.conv2 = conv1, bn1, conv2

    before_finetuning, _ = test_model(structured, x_test, y_test, "cpu")
    print(f"Structured accuracy before fine-tuning: {before_finetuning}%")
    fit_model(
        structured,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=6,
        learning_rate=0.01,
    )
    accuracy, _ = test_model(structured, x_test, y_test, "cpu")
    print(f"Structured accuracy after fine-tuning: {accuracy}%")

    torch.save(
        unstructured.state_dict(),
        SAVE_DIR / "unstructured_weights.pth",
    )
    torch.save(structured, SAVE_DIR / "structured_full_model.pth")
    sample = x_test[:1]
    for name, model in (
        ("unstructured", unstructured),
        ("structured", structured),
    ):
        model = model.cpu().eval()
        torch.jit.save(
            torch.jit.trace(model, sample),
            SAVE_DIR / f"{name}.ts",
        )
        torch.onnx.export(
            model,
            sample,
            SAVE_DIR / f"{name}.onnx",
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
            opset_version=17,
            dynamo=False,
        )
    print("Saved pruning checkpoints, TorchScript, and ONNX exports.")


if __name__ == "__main__":
    main()
