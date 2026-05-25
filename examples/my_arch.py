"""Example user spec file.

Generate diagrams for these blocks with::

    python ../_generate.py --specs my_arch.py --out ./out --depth 1
    # or, to skip regenerating the built-in library:
    python ../_generate.py --specs my_arch.py --out ./out --no-builtins
"""

from _generate import _io, _op, _ref

CATEGORY = "myarch"
CATEGORY_DESC = "Worked example: small transformer wired from built-in blocks."

BLOCKS = {
    "TinyTransformerLM": (
        "Token embedding → 2× TransformerEncoderBlock → tied linear head.",
        "(B, T) → logits:(B, T, V)",
        [
            [_io("ids  (B, T)")],
            [_ref("TokenEmbedding")],
            [_ref("LearnedPositionalEmbedding")],
            [_ref("TransformerEncoderBlock")],
            [_ref("TransformerEncoderBlock")],
            [_op("tied linear  (D → V)")],
            [_io("logits  (B, T, V)")],
        ],
    ),
    "MyDiffusionUNet": (
        "Time-conditioned UNet that emits noise predictions.",
        "x_t:(B, C, H, W), t:(B,) → ε̂:(B, C, H, W)",
        [
            [_io("x_t  (B, C, H, W)"), _io("t  (B,)")],
            [_op("image stem"), _ref("SinusoidalTimeEmbedding")],
            [_ref("UNet")],
            [_io("ε̂  (B, C, H, W)")],
        ],
    ),
}
