"""Fixed real-infrared detector. Not trained in this checkout.

The detector is trained once on real LLVIP infrared and then frozen.
It is evaluated on real test infrared and on synthetic infrared from the
same RGB test frames. That measures whether the translation preserves
features a real-infrared detector uses. It does not measure whether
synthetic images are useful as extra detector training data.
"""


def status() -> dict:
    return {
        "id": "plp-detector-real-ir",
        "status": "pending",
        "architecture": "torchvision Faster R-CNN, MobileNetV3-Large FPN",
        "trained_on": "real LLVIP infrared from train_100",
        "tested_on": ["real_ir", "diffusion_100", "diffusion_20", "diffusion_05", "diffusion_01"],
        "metric": "pedestrian AP at IoU 0.5",
        "reason": "Training has not been run. No detection score exists.",
    }
