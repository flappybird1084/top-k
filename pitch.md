# Top-Kernel — Demo video

Production notes: 500 spoken words, approximately 3–4 minutes. Italicized screen directions are not spoken. Present the HTML walkthrough first, then the recorded demo. Keep W&B signed in and the linked ARIA report ready in another tab.

*Show `pitch-v2.html` → Repository. Hold on the repo and dataset fields.*

Training a model means choosing an architecture, an optimizer, a learning rate, and the kernels that execute it. Exploring those choices takes time and compute. This is Top-Kernel: agents that search for better training recipes and faster kernels, with measurements you can inspect.

Let me show you how it works, then walk through the product.

*HTML → Adapter. Expand the failed attempt, then the verified attempt.*

We start with a repository and dataset. The adapter agent connects the model, data loader, and loss to a shared interface. Here, the walkthrough shows a shape mismatch. The error returns to the writing agent, which revises its code. Before search begins, checks confirm that the model builds, data loads, and the loss supports gradients.

*HTML → Research & plan → Coding agents. Select a candidate, then Evaluation.*

The planner assigns experiments using the baseline and previous results. A research agent can investigate prior work. Coding agents implement candidates in parallel, while full training evaluations share the GPU. The curator turns each generation into lessons for the next.

Agents propose changes; the harness measures them. Architecture candidates compete on validation loss under matching training budgets. Kernel candidates must pass correctness checks and improve the full training step against compiled PyTorch.

*Switch to `http://127.0.0.1:8767/?demo=1`. Paste `https://github.com/karpathy/nanoGPT`, select Both, submit, then choose a dataset.*

Now, the product. I paste nanoGPT, select both search modes, and choose a dataset. This recorded demo previews the intake and search experience. The results we will see come from previous benchmark runs, rather than a new nanoGPT training job.

*Let the 26-second diagram replay finish. Follow the growing generations and final selection blocks; allow a brief silent hold.*

Watch the diagram grow. Architecture and kernel orchestrators sit in the center. Each candidate gets its own block. Spinners show work in progress; color distinguishes retained and dropped candidates. New generations appear as earlier evaluations finish, and the final selections connect back to their lineage. We compress that recorded process into seconds here.

*Console → Architecture, then Kernels. Show performance, marimo measurements, and populated traces. Click Final results.*

W&B gives us the experiment record. Weave lets us follow the agent calls behind a candidate, and marimo exposes the evaluation measurements. Notice that the console is already populated: only the diagram is replaying. We can pause the video here and inspect the recorded work, rather than asking you to trust an animation or a single number on a slide.

The console brings performance graphs, marimo measurements, and agent traces together. From here, we open final results: architecture quality and kernel speed, side by side, measured on separate workloads.

The architecture benchmark reaches 7.21% lower validation loss at an equal 300-second training budget. Its winning recipe combines parallel blocks, reduced KV projections, and a cyclic learning rate. The 15.83% screening improvement uses a shorter, 120-second budget.

For kernels, Astra achieves 3.30% lower full training-step time across four paired measurements. Every pair improves by at least 3.11%, and eight full-state correctness checks pass.

*Return to HTML → ARIA review. Open View W&B report; show the linked report. Return and show the CUDA OOM run and failure trace links.*

Finally, ARIA helps us understand the evidence. We used it in W&B to analyze runs, generate reports, and investigate a controlled CUDA out-of-memory failure. Here is the linked report, alongside the failed run and its trace.

*Finish on the demo’s side-by-side final results.*

Top-Kernel turns optimization into a visible experimental loop: propose, build, measure, learn. You can follow the search, inspect the failures, and trace a result back to its evidence. Bring your repository. Leave with a measured next step.
