# Async GRPO (NCCL Weight Sync Version)

Fast and scalable Ray-Based Asynchronous implementation of [Group Reward Policy Optimization](https://arxiv.org/pdf/2402.03300).

This codebase is a ground-up refactor of the original Async-GRPO project.  
We rebuilt the orchestration layer so that trainer, logprob and rollout workers can exchange weights via **direct NCCL collectives** instead of the Ray object store path that shipped in the initial release (and that was previously listed in the async-grpo README under “Coming soon”).

## Difference From Original Async-GRPO

- **NCCL-based weight sync** from trainer to generation/logprob workers (no Ray object store, RDMA-friendly).
- **Single orchestrator** now registers all worker roles and handles weight propagation and evaluation scheduling centrally.

## High Performance and Accurate

At the time of writting (april 7th 2025), we benchmarked our library against verl and trl and got a 40% throughput improvement against [Verl](https://github.com/volcengine/verl) and >10x against [TRL](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L98). For more information and steps to reproduce, please read the detailed blog post [here](https://ai-innovation.team/blog/async-grpo-blog).

NOTE: experiments done on 2 8xH100 nodes, we used the same setup as [DeepScaleR](https://github.com/agentica-project/deepscaler/) to test our library on both accuracy and throughput.

![](figs/Steps-per-Hour.svg)

We also ran the DeepScaleR setup for the first 200k steps and got a matching reward plot to prove our library works as intended.

![](figs/experiment_plots.svg)

## Introduction

With the advent of reasoning models and inference time scaling generation lengths are dramatically increasing, creating a huge bottleneck for reinforcement learning in general and for GRPO in particular. Current implementations of GRPO such as [HuggingFace's](https://github.com/huggingface/trl/blob/main//docs/source/grpo_trainer.md) or [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF?tab=readme-ov-file) lead to high GPU idle time by synchronously alternating between generation and training.

To attend to these needs we have created _*Async-GRPO*_. This library allows practioners to flexibly scale and independently schedule training and inference across multiple GPUs (regardless of number of nodes) while asynchronously going through the three main stages of GRPO: 1. Actor Roll out Generation, 2. Reference Log Probabilities Inference and 3. Actor training.

![](figs/async-grpo.drawio.svg)

The main innovation is the ability to start training as soon as a minibatch is ready and automatically do gradient accumulation over a whole batch.

The codebase is designed to be extremely modular and easy to hack, handling workload using Ray workers with simple custom load balancing workers.

### Features

- Built-in verifier
- Uses FSDP via Accelerate for model sharding.
- Padding free sample packing based on Flash Attention.
- NCCL-based weight synchronization from the trainer to logprob and generation workers (no Ray object store involvement, works over RDMA).
- Loss computed as the mean across ALL samples across ALL GPUs, and not mean the of GPU-based means: this avoid possible biases where one GPU consistently get a small number of samples compared to others.
- Chunks the logit/loss computation to reduce memory spikes during training - can do full training 70b+ models on a single 8xA100 node.

### Coming Soon

- PyPI package releases.
- Improved delayed load balancing for generation efficiency: send requests to each worker when their current load goes bellow 2x their max concurrent capacity.
- Tensor parallel VLLM workers for long CoTs on large models >=32Billion parameters.
- Improved logging and visualizations.

## Getting Started

All distributed roles (trainer, generation, logprob) still execute as Ray actors, but they are created **inside the Orchestrator actor** rather than being launched manually. Follow these steps to run on one or more nodes:

1. **Environment**

   - Python 3.11+
   - Install requirements on every node that will host any worker:
     ```bash
     pip install -r training/new_framework_based/new_framework/requirements_base.txt
     pip install -r training/new_framework_based/new_framework/requirements_fsdp.txt
     pip install -r training/new_framework_based/new_framework/requirements_vllm.txt
     ```
   - Make sure the project files (models, datasets, this repo) are available on every node (shared filesystem or replicated checkout).

2. **Bring up a Ray cluster**

   On the head node:

   ```bash
   ray start --head --port=9339 --dashboard-host=0.0.0.0
   ```

   On each additional node:

   ```bash
   ray start --address=<HEAD_IP>:9339
   ```

   Verify the cluster with `ray status` or the dashboard.

   **Example: single-node / multi-GPU**

   ```bash
   # on the only node (8 GPUs shown as example)
   ray start --head --port=9339 --num-gpus=8 --dashboard-host=0.0.0.0
   ```

   Once this command succeeds, skip straight to step 4 and launch the orchestrator on the same machine; Ray will place all trainer/logprob/generation workers on the local GPUs.

   **Example: two-node cluster (head + worker)**

   ```bash
   # head node (10.0.0.18) with 8 GPUs
   ray start --head --port=9339 --dashboard-host=0.0.0.0 --num-gpus=8

   # worker node (10.0.0.21) with 8 GPUs
   ray start --address=10.0.0.18:9339 --num-gpus=8
   ```

   Repeat the worker command on every additional node. When you later launch the orchestrator (step 4), it will automatically schedule placement groups across all 16 GPUs; no per-node processes or extra launch scripts are required.

3. **Prepare a config dict**

   Required highlights (see example below):

   - `model_name_or_path`, `data_path`, `test_data_path`: HF model checkpoint and dataset locations.
   - `num_training_gpu_workers`, `num_logprob_gpu_workers`, `num_generation_gpu_workers`: number of single-GPU workers spawned for each role. For trainers this value is the **FSDP world-size** (i.e., the number of `TrainerGPUWorker` ranks managed by the single `TrainerWorker` controller). For logprob/generation it determines how many GPU processes are created in total (see TP/FSDP notes below).
   - `generation_tp_size`, `logprob_fsdp_size`: tensor-parallel degree for vLLM rollout workers and per-group FSDP degree for logprob workers. You can run multiple logprob groups by setting `num_logprob_gpu_workers` to a multiple of `logprob_fsdp_size`. The trainer always has exactly one controller (`TrainerWorker`) but its FSDP size equals `num_training_gpu_workers`.
   - `train_minibatch_sample_size`, `samples_per_question`, `batch_size`, `learning_rate`, `lr_scheduler`, etc.: training hyperparameters (how many rollouts per prompt, how many questions per macro-batch, optimizer schedule, etc.).
   - Token/engine limits: `max_generation_tokens`, `max_tokens_per_gpu`, `max_num_seqs`, `vllm_engine_preset`.
   - Logging/checkpoint metadata: `output_dir`, `wandb_project`, `wandb_entity`, `wandb_run_name`.

   Example JSON file:

   ```json
   {
     "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
     "data_path": "/mnt/data/math_train.jsonl",
     "test_data_path": "/mnt/data/math_eval.jsonl",
     "output_dir": "/mnt/checkpoints/math_run",
     "reward_fns": ["gsm8k"],
     "num_training_gpu_workers": 1,
     "num_generation_gpu_workers": 1,
     "num_logprob_gpu_workers": 1,
     "generation_tp_size": 1,
     "logprob_fsdp_size": 1,
     "samples_per_question": 16,
     "batch_size": 8,
     "train_minibatch_sample_size": 128,
     "learning_rate": 1e-6,
     "lr_scheduler": "constant",
     "num_epochs": 1,
     "eval_every_n_macro_batches": 1,
     "max_generation_tokens": 4096,
     "max_tokens_per_gpu": 4000,
     "max_num_seqs": 512,
     "vllm_engine_preset": "throughput",
     "wandb_project": "MATH_GRPO",
     "wandb_entity": "example-user",
     "wandb_run_name": "qwen2-nccl-test"
   }
   ```

   GPU sizing quick-reference:

   - **All ones (1,1,1,1,1)** → `num_training_gpu_workers = num_logprob_gpu_workers = num_generation_gpu_workers = 1`, `logprob_fsdp_size = generation_tp_size = 1`. The orchestrator spawns three GPU processes in total (trainer/logprob/vLLM).
   - **Two-per-role shorthand (2,2,2,2,2)** → Think of it as `(trainer FSDP size, logprob controller count, logprob FSDP size, generation controller count, generation TP size)`. To realize it, set `num_training_gpu_workers = 2`, `logprob_fsdp_size = generation_tp_size = 2`, and request two controllers for each of logprob/generation (`num_logprob_gpu_workers = 2 * logprob_fsdp_size = 4`, `num_generation_gpu_workers = 2 * generation_tp_size = 4`). Total GPUs = `2 (trainer) + 4 (logprob) + 4 (generation) = 10`. This example makes it explicit how controller counts multiply with per-controller world size.

### Checkpoints & Output Artifacts

- `output_dir` (required in the config) is the root directory for every artifact the orchestrator writes during training.
- Whenever `min_samples_per_checkpoint` samples have been processed since the previous checkpoint, the orchestrator creates a subdirectory `hf_format/step_<global_step>_samples_<total_samples>` inside `output_dir`. Each of these directories contains:
  - Model weights saved via `TrainerGPUWorker.save_model` (HF `save_pretrained` format, sharded if FSDP is used).
  - Optimizer and LR scheduler state stored by `TrainerGPUWorker.save_optimizer_state`.
- The orchestrator’s own counters (`global_step`, `total_samples_seen`, etc.) are persisted as `training_state.json` in the root of `output_dir`, enabling resumable runs by setting `resume_from_path` to any previous checkpoint directory.
- Only `keep_last_n_checkpoints` of the newest checkpoint folders are kept (default `2`). Older ones are automatically pruned.
- To adjust checkpointing behavior:
  - Increase/decrease `min_samples_per_checkpoint` to control how frequently checkpoints are written.
  - Change `keep_last_n_checkpoints` to retain more or fewer historical checkpoints on disk.
- The “final weights” of a run are simply the newest directory under `hf_format/`. Point inference jobs—or a future run’s `resume_from_path`—at that directory to load the trained model.

4. **Launch the Orchestrator (head node or any machine that can reach Ray)**

   Create a small script, e.g. `launch_orchestrator.py`:

   ```python
   import json
   import ray
   from orchestrator import Orchestrator

   with open("config_math_example.json") as f:
       cfg = json.load(f)

   ray.init(address="auto", namespace="async-grpo")
   orchestrator = Orchestrator.options(
       name="orchestrator",
       namespace="async-grpo",
       lifetime="detached",
   ).remote(config_dict=cfg)
   orchestrator.run.remote()
   ```

   Then run:

   ```bash
   python launch_orchestrator.py
   ```

   The Orchestrator actor allocates placement groups for trainer/logprob/rollout workers across the cluster, initializes NCCL collectives, and starts the async training loop. Submitting this script through your job scheduler (or keeping it in a tmux/screen session) is sufficient.

   > **Multi-node tip:** you only run the orchestrator once per training run. After `ray.init(address="auto", ...)` it will discover every GPU registered with the Ray cluster (across all nodes) and place workers there; there is no per-node launcher beyond the `ray start` commands from step 2.

5. **Monitoring / shutdown**

   - Use the Ray dashboard or `ray list actors`/`ray status` to monitor worker health. Logs for each worker are under Ray’s logging directory and the Orchestrator streams WandB metrics if configured.
   - To stop the run, either terminate the Python script (the detached actor will finish its outstanding work) or explicitly kill it:
     ```python
     ray.kill(orchestrator)
     ```
     Afterward, run `ray stop` on each node when you are done with the cluster.

## Customization

### Data Input

We expect passed-in data to be in a JSONL format, with two required fields:

- `input_token_ids`: The token ids for the desired input prompts to be used during rollout generation. This should be your final, processed prompt(with chat template applied if necessary).
- `input`: The string input for the desired prompts to be used during rollout generation, essentially the decoded version of `input_token_ids`. Note that this field name is only strictly necessary when using our default verifier. A custom verifier could also reference any custom field name.
- `answer`: The ground truth answer to be compared against in the reward function / verifier. Note that this field name is only strictly necessary when using our default verifier. A custom verifier could also reference any custom field name.

### Custom Reward Functions

Async-GRPO now supports fully pluggable reward adapters via `reward_registry.py` and the `RewardType` enum. No core code changes are needed to introduce new reward logic.

1. Implement your adapter function in `reward_registry.py`:

   ```python
   def my_custom_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
       """Compute and return at least {'reward': float}, optionally 'format_reward'."""
       # e.g., sample processing...
       reward_value = ...
       format_value = ...  # optional
       return { 'reward': reward_value, 'format_reward': format_value }
   ```

2. Register it under a new enum member:

   ```python
   class RewardType(str, Enum):
       MATHD = "mathd"
       SYMPY = "sympy"
       COUNTDOWN = "countdown"
       MY_CUSTOM = "my_custom"

   REWARD_ADAPTERS[RewardType.MY_CUSTOM] = my_custom_adapter
   ```

3. Reference the adapter in your orchestrator config:

   ```json
   {
     "reward_fns": ["mathd", "my_custom"]
   }
   ```

   Restart the orchestrator (or reload the config) so the ExperienceManager and vLLM workers pull in the updated reward list.

The `VerifierPool` will spawn one Ray worker per adapter, run them in parallel on each sample, and select the best result (highest `reward` or any successful computation). This makes it trivial to mix and match or extend reward functions without touching the pool or worker logic.

# Architecture Explanation

#### System Overview

Ray still provides the distributed substrate, but now a single [`Orchestrator`](orchestrator.py) actor is responsible for:

- Spawning placement groups for trainer/logprob/generation worker pools
- Building the NCCL collectives that connect every GPU (trainer + inference)
- Coordinating experience flow, evaluation, checkpointing, and WandB logging

All other actors register back to this orchestrator so no external registry scripts are required.

#### Sample Data Structure

The unit data structure is the concept of a `sample`. It is a dictionary that should contain `input_token_ids` (the [sample dataset](math_simplerl_qwen_data_token_ids.jsonl) contains it). Thus, every
worker processes one unit of data at a time but asynchronously. Each “sample” is a dict containing:

- `input_token_ids` / `input` – the prompt (chat template already applied)
- `answer` – ground-truth for verifiers/reward adapters
- Metadata produced during rollout (UIDs, sample ids, advantages, etc.)

Samples flow through generation → logprob → training entirely asynchronously. See `sample_processing_utils.py` for packing details (always flattened 1×L tensors in this version).

#### Generation Workers

[`GenerationVLLMWorker`](vllm_worker.py) owns a vLLM `AsyncLLM` engine (one per TP group). Responsibilities:

- Generate `samples_per_question` rollouts for each prompt
- Run verifier adapters (MathD/Sympy/GSM8K etc.) to compute rewards/advantages
- Maintain per-worker NCCL group membership so the orchestrator can broadcast trainer weights directly over NCCL without Ray object stores

Under the hood the individual GPU processes that actually host the vLLM model live in `gpu_process_worker.py` as `RolloutGPUWorker` instances. `GenerationVLLMWorker` is a Ray “controller” actor that groups those 1-GPU workers via tensor parallelism and exposes a single inference interface back to the orchestrator.

#### Logprob Workers

[`LogProbGPUWorker`](gpu_process_worker.py via `LogProbFSDPWorker` controller) mirrors the trainer stack:

- Loads the same HF model, applies the GRPO forward shim (`grpo_loss.py`)
- Packs incoming rollouts with `post_process_batch` and computes per-token logprobs/entropies
- Participates in the same NCCL collectives as trainers to receive weight broadcasts

Just like the generation path, there are two layers: `LogProbGPUWorker` (single-GPU FSDP rank defined in `gpu_process_worker.py`) and the controller `LogProbFSDPWorker` (from `logprob_worker.py`) that manages the FSDP group, performs collective operations, and exposes a simplified RPC surface to the orchestrator.

#### Trainer Workers

[`TrainerGPUWorker`](gpu_process_worker.py) handles FSDP training:

- Consumes microbatches from the `ExperienceManager`
- Computes GRPO loss/metrics, performs gradient accumulation, and applies optimizer steps
- After each macro-batch, summarizes metrics for the orchestrator and triggers NCCL weight syncs toward inference workers

On top of the GPU-level workers we run `TrainerWorker` (from `trainer_worker.py`), a Ray controller that holds the list of trainer ranks, drives FSDP initialization, and provides helper RPCs (e.g., fetch state dict, broadcast sync instructions). The pattern is therefore:

```
TrainerWorker (controller) <-> TrainerGPUWorker (single GPU, FSDP rank)
LogProbFSDPWorker (controller) <-> LogProbGPUWorker (single GPU, FSDP rank)
GenerationVLLMWorker (controller) <-> RolloutGPUWorker (single GPU, TP rank)
```

This separation keeps GPU-bound code in `gpu_process_worker.py` while higher-level orchestration stays in the controller actors.

> **Controller counts:** the orchestrator always spawns exactly one `TrainerWorker` controller (it manages `num_training_gpu_workers` FSDP ranks). For logprob and generation we can have multiple controllers: each logprob controller owns `logprob_fsdp_size` GPUs, so total logprob GPU workers = `num_logprob_gpu_workers`. Each generation controller owns `generation_tp_size` GPUs, so total rollout workers = `num_generation_gpu_workers`. Choose those numbers to match your cluster layout (see the config section for details).

#### Experience Manager

[`ExperienceManager`](orchestrator.py) is a dedicated actor created by the orchestrator. It:

- Tracks all pending generation/logprob tasks for the current macro-batch
- Builds microbatches that respect per-GPU token limits
- Signals trainers when to run microbatch, gradient-step, and batch-done phases
- Aggregates generation/logprob timing + reward stats for logging

#### Weight Synchronization (NCCL)

`Orchestrator.sync_all_weights()` chooses the trainer rank-0 actor as source, then uses `ray.util.collective` to broadcast parameter shards (and metadata) to every trainer/logprob/generation worker. This replaces the old Ray object store path and uses RDMA where available.

#### Training Loop

The orchestrator’s `_run_training_loop` drives the entire async GRPO pipeline:

1. Load dataset via Hugging Face `datasets`, attach `__uid__`, build PyTorch dataloader
2. For each macro-batch:
   - Send prompts to generation/logprob pools via `ExperienceManager`
   - Collect trainer metrics/completions (microbatch + minibatch granularity)
   - Log metrics to WandB, run evaluations/checkpoints per config
   - Broadcast latest weights via NCCL

The flow is fully asynchronous: trainers never block on generation as long as new samples arrive, and inference workers are free to run on separate nodes with their own GPU pools.

#### Ray Utilities & NCCL Groups

`ray_utils.py` contains the helper abstractions (`RayResourcePool`, `RayClassWithInitArgs`, `RayWorkerGroup`) that the Orchestrator uses to:

- Allocate placement groups with the right number of GPUs/CPUs per worker type
- Spawn controller actors alongside their single-GPU workers
- Collect the worker handles needed to form the global NCCL group

Without these utilities, building a single NCCL communicator that spans trainer + logprob + generation ranks (potentially across nodes) would be extremely cumbersome. They give us the guarantees needed to call `collective.create_collective_group` once per run and keep every GPU in lock-step for weight broadcasts.

#### GRPO Loss Details

Our GRPO loss is mathematically equivalent to that described in the original [GRPO paper](https://arxiv.org/pdf/2402.03300) — specifically, equation 19 (further simplified by having $\pi_{\theta_{\text{old}}} = \pi_\theta$, as the policy isn't updated more than once per batch, aligning with equation 20's gradient).

To compute this loss:

1. **Batch Postprocessing:**  
   We use [post_process_batch](sample_processing_utils.py) to pack all samples into a single-dimensional `input tensor` for flash-attention, padding-free training (see [this](https://huggingface.co/blog/packing-with-FA2)). This function also produces an `output_indices` tensor indicating output token positions and broadcasts constants (like each sample's advantage and output length) across the batch.

2. **Log Probability Computation:**  
   With the postprocessed batch, we compute per-token log probabilities by slightly modifying the typical cross-entropy loss function ([`PerTokenLogProbsFromCE`](grpo_loss.py)). This leverages that the log probability of a token is equivalent to the negative cross-entropy loss for that token. Labels are set to \(-100\) for all tokens except the output tokens, ensuring that non-output tokens contribute 0 to the loss.

3. **Policy Gradient Loss:**  
   We compute the per-token policy gradient loss using: $L_{pg} = -A_i \log \pi_\theta(a_t|a_{1:t-1})$  
   This serves as a per-token Taylor expansion approximation of the KL divergence (approximately $\pi_{\text{ref}}/\pi_\theta - \log(\pi_{\text{ref}}/\pi_\theta) - 1$). The losses are divided by the number of tokens in the output (as the RL loss is computed at the trajectory level) and summed across all samples (without averaging).

4. **Scaling Adjustments:**  
   The loss is scaled up by the number of training processes to counteract FSDP's mean reduction, and the gradients are scaled down by the total number of samples (across all training processes) to effectively average across the batch.

> ### Original Async-GRPO Reference
>
> The following snippets come from the _original_ async-grpo project (pre-NCCL refactor). They are **not compatible** with the NCCL weight-sync architecture described in this README; keep them only if you need to inspect or reproduce the legacy workflow as-is.
>
> #### Prepare the dataset
>
> This is custom for the task you're training on. In the original release they used the [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) dataset:
>
> ```bash
> cd sample-data
> python create_count_down.py
> cd ../
> ```
>
> #### Start the training on the nodes you want to use for training
>
> Example DeepScaleR reproduction command:
>
> ```bash
> CUDA_VISIBLE_DEVICES=2,3 torchrun \
>   --nnodes=1 \
>   --node_rank=0 \
>   --nproc_per_node=2 \
>   --rdzv_id=101 \
>   --rdzv_endpoint=localhost:54367 \
>   trainer_core.py \
>     --model-name-or-path         Qwen/Qwen2-1.5B-Instruct \
>     --learning-rate              2e-6 \
>     --batch-size                 128 \
>     --lr-scheduler              constant_with_warmup \
>     --num-warmup-steps           5 \
>     --max-tokens-per-gpu         30000 \
>     --samples-per-question       32 \
>     --temperature                1.0 \
>     --max-generation-tokens      16384 \
>     --data-path                  sample-data/count_down_tasks_3to4.jsonl \
>     --min-samples-per-checkpoint 20000 \
>     --output-dir                 dummy_output/ \
>     --dump-samples-filename      dummy_output/dump_samples.jsonl \
>     --infinite-sampler-seed      223 \
>     --train-minibatch-size       32 \
>     --num-training-batches       1000000 \
>     --logging-level              INFO \
>     --use-torch-compile \
>   2>&1 | tee train.log
> ```
>
> #### Plotting training metrics
>
> ```bash
> python plot.py plot <PATH/TO/training_metrics.jsonl> \
>   --remote YOUR_SSH_ALIAS \
>   -n YOUR_EXPERIMENT_NAME \
>   -v avg_reward \
>   -v avg_max_reward_in_group \
>   -v avg_output_tokens \
>   -v entropy \
>   -v perc_truncated_samples \
>   -v perc_with_0_advantage \
>   -x total_samples_accumulated \
>   -s --smooth-window 40 \
>   -o experiment_plots.svg
> ```
>
> If your `training_metrics.jsonl` file is already local:
>
> ```bash
> python plot.py plot path/to/training_metrics.jsonl \
>   -n my_experiment \
>   -v avg_reward -v entropy \
>   -x total_samples_accumulated \
>   -o experiment_plots.svg
> ```
>
> #### Troubleshooting notes from the original project
>
> - When a Ray worker fails, the driver (the process that spawns such worker) may print unrelated errors. Check worker logs directly.
> - Sometimes errors happen and Ray logging handler fails to show the tracebacks, failing silently, debug printing to a file is recommended to enable visibility.
> - When things fail, do `ray stop` everywhere and restart the process on all nodes. Ray becomes a bit unstable when restarting processes. Also delete the ray temp directory to avoid any issues.
> - It's important to create a separate conda environment for the training process or the worker environments will become corrupted. The python version should be the same as the base environment.
> - `ray list actors | grep ALIVE` can be used to check if all the expected workers are running.
> - Make sure you can do enough HTTP connections on your cluster: `ulimit -n 65535`
> - Ray creates a lot of temporary files in the `/tmp` directory. You can clean them up with `rm -rf /tmp/ray`. Also, you need enough space, otherwise use `ray start --temp-dir=/dev/shm/ray` to use the shared memory as a temporary directory.
> - When a worker fails, it's better to replicate the environment and run it without Ray runtime management. For example, if the logprob worker fails:
>   ```bash
>   conda create -n grpo_logprob python=3.12 -y
>   conda activate grpo_logprob
>   pip install -r requirements_base.txt
>   pip install -r requirements_fsdp.txt
>   torchrun --nproc_per_node=1 logprob_worker.py --model_path Qwen/Qwen2-1.5B-Instruct 2>&1 | tee logprob_worker_1.log
>   ```
