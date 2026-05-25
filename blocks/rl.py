"""Reinforcement-learning building blocks."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "rl"
CATEGORY_DESC = "Reinforcement-learning building blocks."

BLOCKS: Dict[str, Spec] = {
    "PolicyNetwork": (
        "Maps a state to an action distribution / Gaussian (μ, σ) / discrete logits.",
        "s:(B, D_s) → π(a | s)",
        [
            [_io("state  (B, D_s)")],
            [_op("MLP trunk")],
            [_op("head → π(a | s)")],
            [_io("action / dist  (B, D_a)")],
        ],
    ),
    "ValueNetwork": (
        "Estimates V(s).",
        "s:(B, D_s) → V:(B,)",
        [
            [_io("state  (B, D_s)")],
            [_op("MLP")],
            [_op("head → V(s)")],
            [_io("value  (B,)")],
        ],
    ),
    "QNetwork": (
        "Estimates Q(s, a) — either by concatenating (s, a) or with one head per discrete action.",
        "s, a → Q:(B,)",
        [
            [_io("state, action  (B, D_s) + (B, D_a)")],
            [_op("concat / one-hot")],
            [_op("MLP")],
            [_io("Q(s, a)  (B,)")],
        ],
    ),
    "ActorCritic": (
        "Shared trunk that branches into a policy head and a value head.",
        "s → π(a | s),  V(s)",
        [
            [_io("state  (B, D_s)")],
            [_op("shared MLP trunk")],
            [_op("policy head"), _op("value head")],
            [_io("π, V  (B, D_a), (B,)")],
        ],
    ),
    "ReplayBuffer": (
        "Ring buffer storing transitions; sampled in mini-batches for off-policy learning.",
        "(s, a, r, s', d) → batch",
        [
            [_io("transition (s, a, r, s', d)  (·,)")],
            [_op("ring buffer  (capacity N)")],
            [_op("uniform / prioritised sample")],
            [_io("mini-batch  (B, ·)")],
        ],
    ),
    "TargetNetwork": (
        "Slowly-tracking copy of the online network used to stabilise bootstrap targets.",
        "θ_online → θ_target",
        [
            [_io("θ_online  (params)")],
            [_op("θ_target ← τ · θ_target + (1 − τ) · θ_online")],
            [_io("θ_target  (params)")],
        ],
    ),
}
