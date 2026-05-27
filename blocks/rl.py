"""Reinforcement-learning building blocks."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "REINFORCE — vanilla policy gradient",
                "A2C / A3C — synchronous & asynchronous actor-critic",
                "TRPO / PPO — trust-region / clipped policy improvement",
                "SAC — entropy-regularised stochastic policy for continuous control",
                "Decision-Transformer policies for offline RL",
            ],
            tasks=[
                "Continuous control (MuJoCo, robotic manipulation, locomotion)",
                "Discrete-action games (Atari, board games, card games)",
                "Dialogue / tool-use policies in LLM agents (RLHF, RLAIF)",
                "Anywhere you need a *stochastic* mapping `state → action`",
            ],
            pitfalls=[
                "High-variance gradients — pair with a value baseline / GAE.",
                "Action squashing (`tanh`) breaks log-probability — use the "
                "tanh-squashed-Gaussian correction.",
                "Stochastic policies can collapse to a deterministic mode; "
                "add an entropy bonus (SAC, max-ent RL).",
                "Discrete-action heads need numerically-stable softmax + log-softmax "
                "for the log-prob (don't compute `log(softmax(x))` naively).",
            ],
            see_also=[
                "[REINFORCE (Williams 1992)](https://link.springer.com/article/10.1007/BF00992696)",
                "[A3C (Mnih et al. 2016)](https://arxiv.org/abs/1602.01783)",
                "[PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)",
                "[SAC (Haarnoja et al. 2018)](https://arxiv.org/abs/1801.01290)",
            ],
        ),
    ),
    "ValueNetwork": (
        "Estimates V(s) — the expected return from a state under the current policy.",
        "s:(B, D_s) → V:(B,)",
        [
            [_io("state  (B, D_s)")],
            [_op("MLP")],
            [_op("head → V(s)")],
            [_io("value  (B,)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "Critic in every actor-critic algorithm (A2C, A3C, PPO, SAC)",
                "Baseline for variance reduction in policy gradients",
                "Bootstrap target for n-step / TD(λ) returns",
                "Generalised Advantage Estimation (GAE)",
            ],
            tasks=[
                "Reducing gradient variance in REINFORCE / PG methods",
                "Computing advantages `A(s, a) = Q(s, a) − V(s)`",
                "On-policy bootstrapping where Q(s, a) is impractical "
                "(continuous action spaces)",
            ],
            pitfalls=[
                "Bootstrapping with the same network used for updates causes "
                "instability — use a slowly-tracking TargetNetwork for the target.",
                "Tightly coupling actor and critic learning rates often "
                "destabilises both; tune them independently.",
                "MSE loss against high-variance returns can over-fit; clip the "
                "value loss (PPO) or use Huber loss (DQN family).",
            ],
            see_also=[
                "[Temporal-difference learning (Sutton 1988)](https://link.springer.com/article/10.1007/BF00115009)",
                "[GAE (Schulman et al. 2015)](https://arxiv.org/abs/1506.02438)",
                "[PPO value-function clipping](https://arxiv.org/abs/1707.06347)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DQN — discrete actions, one head per action",
                "Double DQN — decoupled action-selection vs target-evaluation",
                "Dueling DQN — V(s) + advantage stream",
                "DDPG / TD3 — Q(s, a) with concatenated state-action input",
                "SAC — twin Q-networks to mitigate maximisation bias",
                "Rainbow — combines DQN, Double, Dueling, PER, n-step, NoisyNets, C51",
            ],
            tasks=[
                "Off-policy value-based learning on Atari / discrete benchmarks",
                "Continuous control via deterministic policy gradient (DDPG/TD3/SAC)",
                "Hybrid imitation + RL (Q-filter for behaviour cloning)",
            ],
            pitfalls=[
                "Maximisation bias from `max_a Q(s, a)` — use Double DQN or twin "
                "Q-networks (TD3, SAC) to decorrelate target selection.",
                "Replay-buffer correlation breaks i.i.d. assumption — randomise "
                "minibatches and stagger updates.",
                "Bootstrapping with the online net diverges easily — always pair "
                "with a TargetNetwork.",
                "Concatenated-action MLPs scale poorly for high-dim discrete "
                "spaces — use per-action heads or factored representations.",
            ],
            see_also=[
                "[DQN (Mnih et al. 2015, Nature)](https://www.nature.com/articles/nature14236)",
                "[Double DQN (van Hasselt et al. 2016)](https://arxiv.org/abs/1509.06461)",
                "[Dueling DQN (Wang et al. 2016)](https://arxiv.org/abs/1511.06581)",
                "[DDPG (Lillicrap et al. 2016)](https://arxiv.org/abs/1509.02971)",
                "[TD3 (Fujimoto et al. 2018)](https://arxiv.org/abs/1802.09477)",
                "[Rainbow (Hessel et al. 2018)](https://arxiv.org/abs/1710.02298)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "A2C / A3C — original sync / async actor-critic",
                "PPO — clipped policy ratio + value loss on the same trunk",
                "IMPALA — distributed actor-critic with V-trace corrections",
                "MuZero — model + policy + value heads on a shared latent",
            ],
            tasks=[
                "Sample-efficient on-policy learning where critic reduces "
                "policy-gradient variance",
                "Large-scale distributed RL where one trunk feeds many workers",
                "RLHF — actor outputs token distributions, critic scores partial "
                "responses for advantage computation",
            ],
            pitfalls=[
                "Policy and value losses compete for trunk capacity — clip the "
                "value loss or weight it (`c_v ≈ 0.5`).",
                "Entropy bonus is essential for exploration in early training; "
                "anneal it as the policy sharpens.",
                "Gradient norm explodes on bad batches — global-norm clip "
                "(typically 0.5–1.0).",
            ],
            see_also=[
                "[A3C (Mnih et al. 2016)](https://arxiv.org/abs/1602.01783)",
                "[PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)",
                "[IMPALA (Espeholt et al. 2018)](https://arxiv.org/abs/1802.01561)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "DQN — original uniform-replay experience buffer",
                "PER — sampling proportional to TD-error magnitude",
                "DDPG / SAC / TD3 — off-policy continuous control",
                "Ape-X / R2D2 — distributed prioritised replay at scale",
                "HER — Hindsight Experience Replay for sparse-reward goals",
            ],
            tasks=[
                "Breaking temporal correlation in updates",
                "Reusing rare / high-value transitions many times",
                "Off-policy correction when the data-generating policy "
                "differs from the current learner",
            ],
            pitfalls=[
                "Buffer stays full of stale data when the policy drifts fast; "
                "tune capacity vs. update ratio.",
                "PER without importance-sampling weights introduces bias — "
                "always anneal β from ~0.4 → 1.0.",
                "Storing full image observations is RAM-heavy — use frame stacks "
                "+ uint8 storage, decode in the worker.",
                "Multi-step returns need transition aggregation (n-step) before "
                "sampling, not after.",
            ],
            see_also=[
                "[Experience replay (Lin 1992)](https://link.springer.com/article/10.1007/BF00992699)",
                "[Prioritised Experience Replay (Schaul et al. 2015)](https://arxiv.org/abs/1511.05952)",
                "[Hindsight Experience Replay (Andrychowicz et al. 2017)](https://arxiv.org/abs/1707.01495)",
                "[Ape-X (Horgan et al. 2018)](https://arxiv.org/abs/1803.00933)",
            ],
        ),
    ),
    "TargetNetwork": (
        "Slowly-tracking copy of the online network used to stabilise bootstrap targets.",
        "θ_online → θ_target",
        [
            [_io("θ_online  (params)")],
            [_op("θ_target ← τ · θ_target + (1 − τ) · θ_online")],
            [_io("θ_target  (params)")],
        ],
        [],
        None,
        _notes(
            used_in=[
                "DQN — hard copy every C steps",
                "DDPG / TD3 / SAC — soft Polyak averaging (`τ ≈ 0.005`)",
                "MuZero — separate target net for the value prediction head",
                "Distillation pipelines — teacher network tracking a moving "
                "average of the student",
            ],
            tasks=[
                "Stabilising the bootstrap target in TD-learning",
                "Decoupling target evaluation from policy improvement to avoid "
                "moving-goalposts divergence",
            ],
            pitfalls=[
                "τ too high → targets move fast → instability (oscillating Q).",
                "τ too low → slow learning, lagging targets.",
                "Forgetting to detach `θ_target` from the autograd graph leaks "
                "gradients into the target — always `with torch.no_grad():`.",
                "Hard copies every C steps cause periodic learning spikes; "
                "soft updates smooth them.",
            ],
            see_also=[
                "[DQN (Mnih et al. 2015, Nature)](https://www.nature.com/articles/nature14236)",
                "[DDPG (Lillicrap et al. 2016)](https://arxiv.org/abs/1509.02971)",
                "[TD3 (Fujimoto et al. 2018)](https://arxiv.org/abs/1802.09477)",
            ],
        ),
    ),
}
