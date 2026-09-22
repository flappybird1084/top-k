'use strict';
/* Repo network for the front-page "web" scene.
   `nanochat` is the one real, measured data point (findings/01-results.md:
   row-parallel Triton RMSNorm, training step 8.95ms -> 8.58ms, -4.1% vs the
   torch.compile incumbent). Every other entry is an illustrative placeholder
   sized for visual variety, not a claimed result — swap in real numbers as
   more repos are run. `verified` distinguishes the two in the UI. */
window.TOPK_REPOS = [
  { id: 'nanochat', name: 'karpathy/nanochat', pct: 4.1, verified: true, note: 'Triton RMSNorm, verified: 8.95ms → 8.58ms training step.' },
  { id: 'nanogpt', name: 'karpathy/nanoGPT', pct: 6.8, verified: false },
  { id: 'llmc', name: 'karpathy/llm.c', pct: 21.7, verified: false },
  { id: 'gptneox', name: 'EleutherAI/gpt-neox', pct: 9.2, verified: false },
  { id: 'llama', name: 'facebookresearch/llama', pct: 12.4, verified: false },
  { id: 'llama3', name: 'facebookresearch/llama3', pct: 14.1, verified: false },
  { id: 'mistral', name: 'mistralai/mistral-src', pct: 8.5, verified: false },
  { id: 'transformers', name: 'huggingface/transformers', pct: 5.9, verified: false },
  { id: 'deepspeed', name: 'microsoft/DeepSpeed', pct: 11.3, verified: false },
  { id: 'megatron', name: 'NVIDIA/Megatron-LM', pct: 13.7, verified: false },
  { id: 'maxtext', name: 'google/maxtext', pct: 10.6, verified: false },
  { id: 'levanter', name: 'stanford-crfm/levanter', pct: 7.4, verified: false },
  { id: 'llmfoundry', name: 'mosaicml/llm-foundry', pct: 16.2, verified: false },
  { id: 'redpajama', name: 'togethercomputer/RedPajama', pct: 9.8, verified: false },
  { id: 'cerebras', name: 'Cerebras/modelzoo', pct: 6.3, verified: false },
  { id: 'megads', name: 'bigscience-workshop/Megatron-DeepSpeed', pct: 12.9, verified: false },
  { id: 'stablelm', name: 'Stability-AI/StableLM', pct: 8.1, verified: false },
  { id: 'falcon', name: 'tiiuae/falcon', pct: 10.1, verified: false },
  { id: 'olmo', name: 'allenai/OLMo', pct: 15.5, verified: false },
  { id: 'mamba', name: 'state-spaces/mamba', pct: 19.3, verified: false },
  { id: 'rwkv', name: 'BlinkDL/RWKV-LM', pct: 11.8, verified: false },
  { id: 'qwen', name: 'QwenLM/Qwen', pct: 7.7, verified: false },
  { id: 'deepseek', name: 'deepseek-ai/DeepSeek-LLM', pct: 13.2, verified: false },
  { id: 'starcoder', name: 'bigcode-project/starcoder', pct: 9.4, verified: false },
  { id: 'codegen', name: 'salesforce/CodeGen', pct: 6.6, verified: false },
  { id: 'dolly', name: 'databrickslabs/dolly', pct: 5.2, verified: false },
  { id: 'openassistant', name: 'LAION-AI/Open-Assistant', pct: 8.9, verified: false },
  { id: 'torchtitan', name: 'pytorch/torchtitan', pct: 17.4, verified: false },
  { id: 'grok', name: 'xai-org/grok-1', pct: 10.9, verified: false }
];
