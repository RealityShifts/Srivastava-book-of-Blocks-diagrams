# ToolUseBlock

> Soft selector over a set of tools, each invoked, results merged back into the state.

**Shapes:** `state → updated state`

```mermaid
%%{init: {'flowchart': {'rankSpacing': 10, 'nodeSpacing': 30}}}%%
flowchart TD
    n0_0["state  (B, D)"]:::io
    n1_0["tool selector  (softmax)"]:::op
    n2_0["Tool 1"]:::op
    n2_1["…"]:::op
    n2_2["Tool K"]:::op
    n3_0["merge tool outputs"]:::merge
    n4_0["new state  (B, D)"]:::io
    n0_0 --> n1_0
    n1_0 --> n2_0
    n1_0 --> n2_1
    n1_0 --> n2_2
    n2_0 --> n3_0
    n2_1 --> n3_0
    n2_2 --> n3_0
    n3_0 --> n4_0
    classDef io fill:#f1f5f9,stroke:#334155,stroke-width:1.4px,color:#0f172a
    classDef op fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.4px,color:#1e3a8a
    classDef norm fill:#dcfce7,stroke:#15803d,stroke-width:1.4px,color:#14532d
    classDef act fill:#ffedd5,stroke:#c2410c,stroke-width:1.4px,color:#7c2d12
    classDef attn fill:#ede9fe,stroke:#6d28d9,stroke-width:1.4px,color:#4c1d95
    classDef merge fill:#fef3c7,stroke:#b45309,stroke-width:1.4px,color:#78350f
    classDef emb fill:#fef9c3,stroke:#a16207,stroke-width:1.4px,color:#713f12
    classDef loss fill:#fee2e2,stroke:#b91c1c,stroke-width:1.4px,color:#7f1d1d
    classDef ctrl fill:#f5f5f4,stroke:#52525b,stroke-width:1.4px,color:#27272a
    classDef ref fill:#e0f2fe,stroke:#0369a1,stroke-width:2px,color:#0c4a6e,stroke-dasharray: 4 2
```

**Used in**

- Toolformer — LM learns when to call APIs via self-supervised tags
- Gorilla, ToolLLM — tool-use fine-tuning of open LLMs
- OpenAI function calling / Anthropic tool use APIs (in spirit)
- ReAct / MRKL agent patterns

**Tasks**

- Agentic LLMs that delegate sub-tasks to calculators, search, code interpreters
- Modular pipelines that compose deterministic tools with neural reasoning

**Common pitfalls**

- Soft routing rarely beats hard selection in deployed systems — most production agents do hard calls with thresholding.
- Tool latency dominates end-to-end response time; pipeline parallelism helps.
- Error propagation: failed tool returns must be turned into tokens the LM can correct from.

**See also**

- [Toolformer (Schick et al. 2023)](https://arxiv.org/abs/2302.04761)
- [ReAct (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)
- [Gorilla (Patil et al. 2023)](https://arxiv.org/abs/2305.15334)
