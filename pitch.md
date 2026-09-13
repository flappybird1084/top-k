# Top-Kernel

Training a model means making hundreds of choices: architecture, optimizer, learning rate, and the kernels underneath. Testing those choices takes engineering time. Top-Kernel turns that work into an agent-driven search, with measured results and a trail you can inspect.

Bring a repository. Find a better training recipe.

Here, we paste nanoGPT and choose a dataset. An adapter agent connects the model, data loader, and loss to a shared training interface. When an attempt fails, the error goes back to the agent to revise its code. The interface checks that the model builds, data loads, and the loss supports gradients before search begins.

Now watch the diagram. Architecture and kernel orchestrators coordinate separate searches. Each block represents a candidate: spinning while it works, green when retained, and marked when dropped. Generations grow outward, preserving the lineage of successful changes. This presentation compresses recorded work into seconds; it is not training a model during the animation.

The planner proposes experiments using the baseline, previous results, and lessons from the curator. It can ask a research agent to investigate prior work. Coding agents implement candidates in parallel; training evaluations share the GPU. The curator carries useful findings forward, so the next generation starts with evidence from the last.

The crucial separation is between proposing an improvement and measuring it. Agents write candidates. The evaluation harness measures them. Architecture candidates compete on validation loss under matching training budgets. Kernel candidates must pass correctness checks and improve the full training step against compiled PyTorch. A promising isolated operation is not enough.

Here are the recorded results, side by side, from separate workloads.

On the 481-million-parameter architecture benchmark, the final recipe combines parallel blocks, reduced KV projections, and a cyclic learning rate. Validation loss falls from 5.8477 to 5.4258: a 7.21% reduction at the same 300-second training budget. Screening reached 15.83% at 120 seconds; the final comparison uses the longer, matched budget.

For kernels, Astra's backward implementations reduce full training-step time by 3.30%, measured as the median of four paired reductions. Every pair improves by at least 3.11%. Eight full-state correctness checks pass. The search respects its five-generation limit.

The console connects those numbers to their evidence: W&B tracks experiments, Weave exposes agent traces, and marimo presents kernel measurements. You can move from a winning result back to the candidate and its evaluation.

ARIA makes that evidence useful. We used it in W&B to analyze training runs, generate reports, and investigate a controlled CUDA out-of-memory failure. The pitch links the failed run and trace. That matters because an optimization system must explain failures as clearly as wins.

The final page brings architecture quality and kernel speed together without pretending they came from one combined experiment. You see what improved, how it was measured, and where to inspect the evidence.

Top-Kernel makes model optimization a repeatable workflow: propose, build, evaluate, learn. Our goal is to give every training team an experimental loop they can follow, challenge, and improve. Bring your code. Leave with a measured next step.
