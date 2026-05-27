"""CNN and vision-specific blocks."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

CATEGORY = "cnn_vision"
CATEGORY_DESC = "CNN and vision-specific blocks."

BLOCKS: Dict[str, Spec] = {
    "InceptionBlock": (
        "Parallel multi-scale convs concatenated channel-wise (GoogLeNet).",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("1×1 Conv"), _op("1×1 → 3×3"), _op("1×1 → 5×5"), _op("3×3 MaxPool → 1×1")],
            [_merge("concat (channel)")],
            [_io("y  (B, C', H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "GoogLeNet (Inception v1) ILSVRC-2014 winner",
                "Inception v2/v3/v4, Inception-ResNet",
                "Xception (later replaced concat-of-branches by depthwise separable)",
            ],
            tasks=[
                "Image classification backbones",
                "Capturing features at multiple receptive fields without enumerating depths",
            ],
            pitfalls=[
                "1×1 bottlenecks before 3×3 / 5×5 are essential — without them the branch is FLOP-explosive.",
                "Channel concatenation grows downstream input channels — plan the next block's compute.",
                "Mostly superseded by ResNet-style depth + width scaling; Inception is rare in 2020s arch.",
            ],
            see_also=[
                "[GoogLeNet (Szegedy et al. 2014)](https://arxiv.org/abs/1409.4842)",
                "[Inception v3 (Szegedy et al. 2015)](https://arxiv.org/abs/1512.00567)",
                "[Xception (Chollet 2017)](https://arxiv.org/abs/1610.02357)",
            ],
        ),
    ),
    "DenseBlock": (
        "Each layer's input is the concatenation of all earlier layers' outputs (DenseNet).",
        "(B, C, H, W) → (B, C + k·L, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("BN-ReLU-Conv  (layer 1)")],
            [_merge("concat with x")],
            [_op("BN-ReLU-Conv  (layer 2)")],
            [_merge("concat with all prev")],
            [_op("…  (L layers)")],
            [_io("y  (B, C + k·L, H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "DenseNet-121/169/201/264 — ImageNet backbone",
                "Tiramisu (DenseNet-based semantic segmentation)",
                "Dense U-Net variants for medical imaging",
            ],
            tasks=[
                "Backbones that need strong feature reuse with relatively few parameters",
                "Tasks with limited data (medical) — implicit deep supervision via skip-everywhere",
            ],
            pitfalls=[
                "Memory-hungry at training: all activations are kept for concatenation — use "
                "shared-memory / checkpointed implementations.",
                "Channel count grows linearly with layers (`C + k·L`); the transition layer 1×1 conv "
                "compresses this back down — never skip it.",
                "Slower than ResNet of the same accuracy on GPUs (memory-bound).",
            ],
            see_also=[
                "[DenseNet (Huang et al. 2016)](https://arxiv.org/abs/1608.06993)",
                "[Memory-Efficient DenseNet (Pleiss et al. 2017)](https://arxiv.org/abs/1707.06990)",
            ],
        ),
    ),
    "SqueezeExcitation": (
        "Channel-wise gating via global pool → bottleneck MLP → sigmoid.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Global AvgPool")],
            [_op("FC down  (C → C/r)")],
            [_act("ReLU")],
            [_op("FC up  (C/r → C)")],
            [_act("Sigmoid")],
            [_merge("· x  (channel scale)")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 6, 0)],
        None,
        _notes(
            used_in=[
                "SENet — ILSVRC-2017 winner",
                "MobileNet-V3, EfficientNet-B0..B7 — SE is built into every block",
                "Detection / segmentation backbones (e.g. SE-ResNeXt)",
            ],
            tasks=[
                "Cheap channel re-calibration for almost any CNN",
                "Adding channel-wise attention without significant FLOP cost (~< 1%)",
            ],
            pitfalls=[
                "Reduction ratio r is a sensitive hyperparameter — too small kills capacity, too large is wasteful.",
                "Sigmoid output multiplied by x can saturate gradients in deep stacks; modern variants use "
                "hard-sigmoid for mobile.",
                "Adds latency on memory-bound hardware despite tiny FLOPs.",
            ],
            see_also=[
                "[Squeeze-and-Excitation Networks (Hu et al. 2017)](https://arxiv.org/abs/1709.01507)",
                "[EfficientNet (Tan & Le 2019)](https://arxiv.org/abs/1905.11946)",
            ],
        ),
    ),
    "CBAM": (
        "Sequential channel attention then spatial attention.",
        "(B, C, H, W) → (B, C, H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Channel Attention  (MLP on max+avg pool)")],
            [_merge("· x")],
            [_op("Spatial Attention  (Conv on max+avg over C)")],
            [_merge("· x")],
            [_io("y  (B, C, H, W)")],
        ],
        [(0, 0, 2, 0), (2, 0, 4, 0)],
        None,
        _notes(
            used_in=[
                "CBAM-augmented ResNet, MobileNet, WideResNet",
                "Object detection / segmentation heads needing both channel and spatial focus",
            ],
            tasks=[
                "Lightweight plug-in attention that captures BOTH 'what' (channel) and 'where' (spatial)",
                "Boosting accuracy on small-to-mid CNNs without changing depth/width",
            ],
            pitfalls=[
                "Order matters — channel-first then spatial is what the paper proves; reversing is worse.",
                "Spatial attention uses a single 7×7 conv on a 2-channel pool; replace with 3×3 only if "
                "input resolution is small.",
                "Mostly subsumed by self-attention in modern hybrid CNN/transformer designs.",
            ],
            see_also=[
                "[CBAM (Woo et al. 2018)](https://arxiv.org/abs/1807.06521)",
                "[BAM (Park et al. 2018)](https://arxiv.org/abs/1807.06514)",
            ],
        ),
    ),
    "SpatialPyramidPooling": (
        "Multi-scale fixed-output pooling, concatenated.",
        "(B, C, H, W) → (B, C·Σbins²,)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("AvgPool 1×1"), _op("AvgPool 2×2"), _op("AvgPool 4×4")],
            [_merge("flatten + concat")],
            [_io("y  (B, C·Σbins²,)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "SPP-Net (He et al. 2014) — variable-size input → fixed feature for FC head",
                "Fast R-CNN's RoI pooling is a per-region SPP",
                "YOLOv3/v4 SPP module (max-pool variant) in the head",
            ],
            tasks=[
                "Decoupling input resolution from a final fixed-size MLP classifier",
                "Multi-scale global context aggregation",
            ],
            pitfalls=[
                "If bin counts are large the flattened feature explodes — use small (1, 2, 3) or (1, 2, 4).",
                "Max-pool vs avg-pool changes behaviour — YOLO uses max for sharpness; classification "
                "tends to use avg.",
            ],
            see_also=[
                "[SPP-Net (He et al. 2014)](https://arxiv.org/abs/1406.4729)",
                "[Fast R-CNN (Girshick 2015)](https://arxiv.org/abs/1504.08083)",
            ],
        ),
    ),
    "FeaturePyramidNetwork": (
        "Build multi-scale feature maps via top-down upsampling + lateral 1×1 connections.",
        "(C3, C4, C5) → (P3, P4, P5)",
        [
            [_io("C3, C4, C5  (B, Cᵢ, H/2ⁱ, W/2ⁱ)")],
            [_op("1×1 lateral on each level")],
            [_op("top-down: upsample + add")],
            [_op("3×3 smooth")],
            [_io("P3, P4, P5  (B, C, H/2ⁱ, W/2ⁱ)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Faster R-CNN with FPN, Mask R-CNN — gold standard detection necks",
                "RetinaNet (FPN + Focal Loss)",
                "Panoptic-FPN, PointRend",
            ],
            tasks=[
                "Multi-scale object detection / segmentation",
                "Any dense-prediction task where small AND large objects must be recovered",
            ],
            pitfalls=[
                "All FPN levels must have the SAME channel count for the top-down add to work (commonly 256).",
                "Aliasing from naïve nearest-upsample — the 3×3 smooth conv after the add is not optional.",
                "PAN-FPN, BiFPN add bottom-up paths — often a free accuracy gain for detection.",
            ],
            see_also=[
                "[FPN (Lin et al. 2016)](https://arxiv.org/abs/1612.03144)",
                "[PANet (Liu et al. 2018)](https://arxiv.org/abs/1803.01534)",
                "[BiFPN / EfficientDet (Tan et al. 2019)](https://arxiv.org/abs/1911.09070)",
            ],
        ),
    ),
    "ASPP": (
        "Atrous Spatial Pyramid Pooling: parallel atrous convs + image pool, concatenated.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("1×1 Conv"), _op("3×3 dil=6"), _op("3×3 dil=12"), _op("3×3 dil=18"), _op("Image pool")],
            [_merge("concat")],
            [_op("1×1 Conv")],
            [_io("y  (B, C', H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "DeepLab v2 / v3 / v3+ semantic segmentation",
                "Real-time segmentation models that need large receptive field at low compute",
            ],
            tasks=[
                "Dense scene parsing where small AND huge objects coexist",
                "Multi-scale context aggregation without down-sampling the feature map",
            ],
            pitfalls=[
                "Dilation rates must be tuned to output stride — 6/12/18 are for OS=16; halve them for OS=8.",
                "Gridding artefacts when dilation grows quickly — visualise the receptive-field overlap.",
                "Image-level pooling branch needs careful upsampling back to feature-map size.",
            ],
            see_also=[
                "[DeepLab v3 (Chen et al. 2017)](https://arxiv.org/abs/1706.05587)",
                "[DeepLab v3+ (Chen et al. 2018)](https://arxiv.org/abs/1802.02611)",
            ],
        ),
    ),
    "PixelShuffleUpsample": (
        "Sub-pixel upsampling: rearrange r² channels into r×r spatial blocks.",
        "(B, C, H, W) → (B, C, r·H, r·W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("Conv → C·r² channels")],
            [_op("PixelShuffle r")],
            [_io("y  (B, C, r·H, r·W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "ESPCN — original sub-pixel CNN for super-resolution",
                "EDSR, RCAN, ESRGAN — modern SR architectures",
                "Diffusion-model decoders / VAE upsampling stages",
            ],
            tasks=[
                "Image super-resolution",
                "Upsampling within a decoder without introducing checkerboard artefacts",
            ],
            pitfalls=[
                "Initialisation matters — sub-pixel layers benefit from ICNR init to avoid checkerboards.",
                "Channel count blows up before the shuffle (C·r²) — memory pressure on large feature maps.",
                "Equivalent to a learned transpose-conv but typically cheaper at the same quality.",
            ],
            see_also=[
                "[Sub-pixel CNN / ESPCN (Shi et al. 2016)](https://arxiv.org/abs/1609.05158)",
                "[Checkerboard Artifacts (Odena et al. 2016)](https://distill.pub/2016/deconv-checkerboard/)",
                "[ICNR init (Aitken et al. 2017)](https://arxiv.org/abs/1707.02937)",
            ],
        ),
    ),
    "DeformableConv2d": (
        "Conv whose sample locations are shifted by a learned offset field.",
        "(B, C, H, W) → (B, C', H, W)",
        [
            [_io("x  (B, C, H, W)")],
            [_op("offset Conv → Δp")],
            [_op("bilinear sample at p + Δp")],
            [_op("Conv on sampled features")],
            [_io("y  (B, C', H, W)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "DCNv1 / DCNv2 — object detection (Faster R-CNN with DCN)",
                "Pose estimation, instance segmentation backbones",
                "Optical flow networks where spatial sampling is irregular",
            ],
            tasks=[
                "Tasks needing geometric flexibility (non-rigid objects, varying scales)",
                "Bridging convolution and attention: sparse, content-aware sampling",
            ],
            pitfalls=[
                "Offsets can drift to absurd locations — clamp or add a regularising loss (DCNv2's modulation).",
                "Pure-PyTorch fallback is slow; production code uses CUDA kernel from torchvision.ops.",
                "Doubles parameters of the layer (offset branch); accuracy gain is biggest in detection.",
            ],
            see_also=[
                "[Deformable Convolution (Dai et al. 2017)](https://arxiv.org/abs/1703.06211)",
                "[DCNv2 (Zhu et al. 2018)](https://arxiv.org/abs/1811.11168)",
            ],
        ),
    ),
    "DeformableAttention": (
        "Attention that samples a small set of keys at learned offsets per query (Deformable DETR).",
        "(B, N, C) → (B, N, C)",
        [
            [_io("x  (B, N, C)")],
            [_op("MLP → reference + sampling offsets")],
            [_op("bilinear sample features")],
            [_attn("attention weights (per head, per point)")],
            [_merge("weighted sum + output proj")],
            [_io("y  (B, N, C)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Deformable DETR — convergence in ~10× fewer epochs than DETR",
                "DINO, DN-DETR, H-DETR — modern detection transformers",
                "3D detection (Deformable 3D-DETR)",
            ],
            tasks=[
                "Detection / segmentation with transformer backbones",
                "Multi-scale feature aggregation in attention with sub-quadratic cost",
            ],
            pitfalls=[
                "Each query samples K points per level — too small (K < 4) hurts recall, too large kills speed.",
                "Reference-point initialisation matters; query anchors are typically tied to encoder positions.",
                "Implementation needs custom CUDA op for full speed (ms-deformable-attention).",
            ],
            see_also=[
                "[Deformable DETR (Zhu et al. 2020)](https://arxiv.org/abs/2010.04159)",
                "[DINO (Zhang et al. 2022)](https://arxiv.org/abs/2203.03605)",
            ],
        ),
    ),
}
