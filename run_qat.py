import copy
import random

import torch
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.quantization import (
    convert,
    disable_observer,
    get_default_qat_qconfig,
    prepare_qat,
)

from pipeline_support import (
    RESULTS_DIR,
    SAVE_DIR,
    fit_model,
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

    model = copy.deepcopy(load_baseline()).cpu().train()
    model.fuse_model(is_qat=True)
    model.qconfig = get_default_qat_qconfig("fbgemm")
    model = prepare_qat(model)

    def schedule(current_model, epoch):
        if epoch >= 1:
            current_model.apply(freeze_bn_stats)
        if epoch >= 2:
            current_model.apply(disable_observer)

    fit_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=3,
        learning_rate=1e-3,
        before_epoch=schedule,
    )
    model = convert(model.cpu().eval())
    accuracy, _ = test_model(model, x_test, y_test)
    torch.save(model.state_dict(), SAVE_DIR / "qat_int8_weights.pth")
    sample = x_test[:1]
    torch.jit.save(
        torch.jit.trace(model, sample),
        SAVE_DIR / "qat_int8.ts",
    )
    torch.onnx.export(
        model,
        sample,
        SAVE_DIR / "qat_int8.onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"QAT accuracy after conversion: {accuracy}%")
    print("Saved QAT checkpoint, TorchScript, and ONNX exports.")


if __name__ == "__main__":
    main()
