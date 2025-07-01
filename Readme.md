# Async GRPO

Fast and scalable Ray-Based Asynchronous implementation of [Group Reward Policy Optimization](https://arxiv.org/pdf/2402.03300).

## High Performance and Accurate

At the time of writting (april 7th 2025), we benchmarked our library against verl and trl and got a 40% throughput improvement against [Verl](https://github.com/volcengine/verl) and >10x against [TRL](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L98). For more information and steps to reproduce, please read the detailed blog post [here](https://ai-innovation.team/blog/async-grpo-blog).

NOTE: experiments done on 2 8xH100 nodes, we used the same setup as [DeepScaleR](https://github.com/agentica-project/deepscaler/) to test our library on both accuracy and throughput.

![](figs/Steps-per-Hour.svg)

We also ran the DeepScaleR setup for the first 200k steps and got a matching reward plot to prove our library works as intended.

![](figs/experiment_plots.svg)

## Introduction

With the advent of reasoning models and inference time scaling generation lengths are dramatically increasing, creating a huge bottleneck for reinforcement learning in general and for GRPO in particular. Current implementations of GRPO such as [HuggingFace's](https://github.com/huggingface/trl/blob/main//docs/source/grpo_trainer.md) or [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF?tab=readme-ov-file) lead to high GPU idle time by synchronously alternating between generation and training. 

To attend to these needs we have created *_Async-GRPO_*. This library allows practioners to flexibly scale and independently schedule training and inference across multiple GPUs (regardless of number of nodes) while asynchronously going through the three main stages of GRPO: 1. Actor Roll out Generation, 2. Reference Log Probabilities Inference and 3. Actor training.

![](figs/async-grpo.drawio.svg)

The main innovation is the ability to start training as soon as a minibatch is ready and automatically do gradient accumulation over a whole batch.

The codebase is designed to be extremely modular and easy to hack, handling workload using Ray workers with simple custom load balancing workers.

### Features:
- Built-in verifier 
- Uses FSDP via Accelerate for model sharding.
- Padding free sample packing based on Flash Attention.
- Don't worry about minibatches: Automatic minibatching and gradient accumulation. Minibatches are built online to keep tokens per GPU at a maximum value. 
- Loss computed as the mean across ALL samples across ALL GPUs, and not mean the of GPU-based means: this avoid possible biases where one GPU consistently get a small number of samples compared to others.
- Chunks the logit/loss computation to reduce memory spikes during training - can do full training 70b+ models on a single 8xA100 node.

### Coming Soon
- PyPI package releases.
- PyNCCL weight update from the actor to the generation and reference workers - currently using Ray Object storage which doesn't use RDMA for communication.
- Improved delayed load balancing for generation efficiency: send requests to each worker when their current load goes bellow 2x their max concurrent capacity.
- Tensor parallel VLLM workers for long CoTs on large models >=32Billion parameters.
- Improved logging and visualizations.

# Getting Started

## Base Installation and Setup

On all nodes being used for any component in the run, begin with the following setup:
```bash
conda create -n grpo_base python=3.12 -y
conda activate grpo_base
pip install -r requirements_base.txt
```

Next, start a ray cluster across the nodes. On the head node, run:
```bash
ray start --head \
--port 6379 
```

and then on each additional node, run:
```bash
conda activate grpo_base
ray start --address=head_node_ip:6379
```

## Starting the Components

### Start the inference workers (both to get rollouts from the policy and to compute logprobs from said rollouts)

Once the Ray cluster has been set up, the next step is to spin up the inference workers (this includes both the vLLM rollout workers and the reference logprob workers). On each node being used for inference, launch inference workers on the desired GPUs. 

For example, for a node with 8 GPUs, and using 7 for generation and 1 for logprob, you would do the following:

##### VLLM inference workers

**NOTE:** do this on each node you want to use for inference.

```bash
export num_inference_workers_in_node=1
export vllm_max_num_seqs=64
export vllm_overhead_seqs=16
export reward_fns=countdown
export base_model_path=Qwen/Qwen2-1.5B-Instruct

for i in $(seq 0 $num_inference_workers_in_node); do
  python worker_dispatcher.py \
    --model_path "$base_model_path" \
    --mode generation \
    --tensor_parallel_size 1 \
    --max_num_seqs "$vllm_max_num_seqs" \
    --overhead_seqs "$vllm_overhead_seqs" \
    --write_failed_generation_samples \
    --reward_fns "$reward_fns" \
    --global_num_verifiers 50 2>&1 | tee ~/grpo_vllm_"$i".log
done
```

##### Logprob inference workers

**NOTE:** do this on each node you want to use for logprob inference.

```bash
export num_workers=1
export base_model_path=Qwen/Qwen2-1.5B-Instruct
export temperature=1.0
export logprob_max_tokens_per_gpu=100000

for i in $(seq 1 $num_workers); do
torchrun --nproc_per_node=1 --master_port=1234$i worker_dispatcher.py \
--model_path $base_model_path \
--mode logprob \
--temperature $temperature \
--max_tokens_per_gpu $logprob_max_tokens_per_gpu 2>&1 | tee ~/grpo_logprob_$i.log
done
```

In our test, we used two nodes, a total of 16 GPUs, 14 for generation and 2 for logprob. you must wait until all the workers are started before starting the training, which is shown by `worker <ID> registered` for each worker. Adjust the number of verifiers, each uses one CPU, make sure your cluster has the capacity.

### Prepare trainer env

```bash
conda create grpo_trainer python=3.12 -y; conda activate grpo_trainer; pip install -r requirements_base.txt;pip install -r requirements_fsdp.txt
```

### Prepare the dataset
this is custom for the task you're training on. In our case, we used the [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) dataset. Which we process doing the following:

```bash
cd sample-data
python create_count_down.py
cd ../
```

this will create a file called `count_down_tasks_3to4.jsonl` in the `sample-data` directory.

### Start the training on the nodes you want to use for training

Finally, the last step is to launch the training on the remaining GPUs. In this example case, we trained with 8 GPUs on a single training node.

For example, in our recent DeepScaleR reproduction, we used:
```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --rdzv_id=101 \
  --rdzv_endpoint=localhost:54367 \
  trainer_core.py \
    --model-name-or-path         Qwen/Qwen2-1.5B-Instruct \
    --learning-rate              2e-6 \
    --batch-size                 128 \
    --lr-scheduler              constant_with_warmup \
    --num-warmup-steps           5 \
    --max-tokens-per-gpu         30000 \
    --samples-per-question       32 \
    --temperature                1.0 \
    --max-generation-tokens      16384 \
    --data-path                  sample-data/count_down_tasks_3to4.jsonl \
    --min-samples-per-checkpoint 20000 \
    --output-dir                 dummy_output/ \
    --dump-samples-filename      dummy_output/dump_samples.jsonl \
    --infinite-sampler-seed      223 \
    --train-minibatch-size       32 \
    --num-training-batches       1000000 \
    --logging-level              INFO \
    --use-torch-compile \
  2>&1 | tee train.log
```

## Plotting Training Metrics

After you run training (e.g. via `trainer_core.py`), your output directory will contain:
- **training_metrics.jsonl**: a JSON Lines file with per-batch metrics. Each line is a JSON object with keys such as `step`, `iteration`, `total_samples_accumulated`, `avg_reward`, `entropy`, etc.
- **training_params.json**: a JSON file listing all the command-line parameters used for this run.

To visualize these metrics, use the provided CLI tool `plot.py`:
```bash
python plot.py plot <PATH/TO/training_metrics.jsonl> \
  --remote YOUR_SSH_ALIAS \
  -n YOUR_EXPERIMENT_NAME \
  -v avg_reward \
  -v avg_max_reward_in_group \
  -v avg_output_tokens \
  -v entropy \
  -v perc_truncated_samples \
  -v perc_with_0_advantage \
  -x total_samples_accumulated \
  -s --smooth-window 40 \
  -o experiment_plots.svg
```

If your `training_metrics.jsonl` file is already local, omit the `--remote` flag:
```bash
python plot.py plot path/to/training_metrics.jsonl \
  -n my_experiment \
  -v avg_reward -v entropy \
  -x total_samples_accumulated \
  -o experiment_plots.svg
```

This command writes an SVG (`experiment_plots.svg`) plotting the specified variables against your training progress.

### Troubleshooting

- when a ray worker fails, the driver (the process that spawns such worker) shows unrelated errors. It's usually a module not found error in the child worker.
- Sometimes errors happen and Ray logging handler fails to show the tracebacks, failing silently, debug printing to a file is recommended to enable visibility.
- when things fail, do ray stop everywhere and restart the process on all nodes. Ray becomes a bit unstable when restarting processes. Also delete the ray temp directory to avoid any issues.
- It's important to create a separate conda environment for the training process or the worker environments will become corrupted. The python version should be the same as the base environment.
- `ray list actors | grep ALIVE` can be used to check if all the expected workers are running.
- make sure you can do enough http connections on your cluster: `ulimit -n 65535`
- Ray creates a lot of temporary files in the `/tmp` directory. You can clean them up with `rm -rf /tmp/ray`. Also, you need enough space, otherwise use `ray start --temp-dir=/dev/shm/ray` to use the shared memory as a temporary directory.
- When a worker fails, it's better to replicate the environment and run it without ray runtime environment management. For example, if the logprob worker fails do this:
```bash
conda create -n grpo_logprob python=3.12 -y; conda activate grpo_logprob; pip install -r requirements_base.txt;pip install -r requirements_fsdp.txt
torchrun --nproc_per_node=1 logprob_worker.py --model_path Qwen/Qwen2-1.5B-Instruct 2>&1 | tee logprob_worker_1.log
```

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

3. Launch inference or generation workers with the desired adapters:
   ```bash
   python worker_dispatcher.py --mode generation --reward_fns mathd,my_custom
   ```

The `VerifierPool` will spawn one Ray worker per adapter, run them in parallel on each sample, and select the best result (highest `reward` or any successful computation). This makes it trivial to mix and match or extend reward functions without touching the pool or worker logic.

# Architecture Explanation

#### System Overview

We use Ray to create a distributed system composed of workers with different roles. The main compute-bound workers are the training workers and the two types of inference workers (generation and logprob).

#### Sample Data Structure

The unit data structure is the concept of a `sample`. It is a dictionary that should contain `input_token_ids` (the [sample dataset](math_simplerl_qwen_data_token_ids.jsonl) contains it). Thus, every worker processes one unit of data at a time but asynchronously.

#### Inference Worker Registration

Both inference workers register themselves with a registry (defined in [`vllm_registry.py`](vllm_registry.py)). The purpose of the vllm registries is to manage requests sent to the separate pools of workers. They relay requests and load balance across the workers by sending the requests to the worker handling the least amount of data at a time. There is one copy of this process for each type (generation and logprob), and it is created only by the first inference worker to request it.

We also use [worker_dispatcher.py](worker_dispatcher.py) to dispatch the workers. This was needed to isolate the python environment of the two types of inference workers.

#### Generation Worker

The generation worker ([`GenerationVLLMWorker`](vllm_worker.py)) is responsible for generating rollouts. It uses a vllm asynchronous engine to generate these rollouts, and then it utilizes HF's [math-verify](https://github.com/huggingface/Math-Verify) to compute a reward (it expects a `gt_answer` key in the sample dict). This worker also completes most of the sample dict, including defining the IDs used for reference logprobs and training (`sample_position_ids`), as well as the `advantages` used in GRPO (or the normalized rewards across a sample's rollouts).

#### Logprob Worker

The logprob worker ([`LogprobWorker`](logprob_worker.py)) computes the log probabilities of the rollouts. It uses the same function as the training process to compute the log probabilities ([`PerTokenLogProbsFromCE`](grpo_loss.py)) and leverages the same utility functions to process the samples into input IDs, position IDs, and labels (e.g. [`get_input_for_logprobs`](sample_processing_utils.py)). It loads the model in the same way as the training process (via [`setup_model`](setup_model.py)) but does not wrap it with FSDP. It also accumulates samples in a batch [`_centralize_inference_requests`] until a maximum number of tokens per GPU is reached to keep the GPUs usage at max.

#### Updating Inference Worker Weights

Both the generation and logprob workers have a method to update their weights. The main training process invokes this method through [`update_vllm_worker_weights`](trainer_core.py) to ensure that both the generation models (actors) and the logprob models (reference) are updated accordingly.

#### Experience Batcher

The main process interfacing with the inference workers is the [`ExperienceBatcher`](vllm_experience_batcher.py). There is only one instance of this process, which is created by the training process. Its responsibilities include:

- **Accumulating Batches:** Gathering batches of samples from each training process and sending them for rollouts and logprob computation ([`get_experience_and_ref_logprobs`](vllm_experience_batcher.py)).
- **Dispatching Minibatches:** Receiving responses from inference workers and distributing minibatches to the training processes ([`_create_batches`](vllm_experience_batcher.py)).
- **Batch Optimization:** Creating batches to maximize the use of GPU token limits while ensuring each training process receives at least one sample ([`add_sample_to_batches`](vllm_experience_batcher.py)).
- **Minimizing Downtime:** Dispatching batches promptly as soon as inference responses are complete.

#### Training Process

The [train](trainer_core.py) script orchestrates the entire workflow:

- **Model Setup and Wrapping:** Sets up the model, wraps it in FSDP, and creates the training objects (e.g., optimizer, learning rate scheduler).
- **Data Loading:** Constructs an infinite sampler for the dataloader, ensuring each rank receives a distinct portion of the dataset.
- **Batch Processing:** Samples batches from the dataloader and sends all samples to the experience batcher. It then asynchronously waits for the batches to be returned with rollouts and logprobs.
- **Loss Computation:** Computes the [GRPO loss](grpo_loss.py), which calculates a per-token loss for each sample in the batch, sums them (without averaging), and scales the loss by the number of training processes (to compensate for FSDP's mean reduction).
- **Gradient Scaling:** Before performing a gradient step, scales the gradients ([`take_gradient_step`](trainer_core.py)) by the total number of samples in the batch across all GPUs, ensuring the mathematical equivalence of processing the entire batch in one forward/backward pass.
- **Model Update:** Executes the gradient step and saves the model to disk.

#### GRPO Loss Details

Our GRPO loss is mathematically equivalent to that described in the original [GRPO paper](https://arxiv.org/pdf/2402.03300) — specifically, equation 19 (further simplified by having $\pi_{\theta_{\text{old}}} = \pi_\theta$, as the policy isn't updated more than once per batch, aligning with equation 20's gradient).

To compute this loss:

1. **Batch Postprocessing:**  
   We use [post_process_batch](sample_processing_utils.py) to pack all samples into a single-dimensional `input tensor` for flash-attention, padding-free training (see [this](https://huggingface.co/blog/packing-with-FA2)). This function also produces an `output_indices` tensor indicating output token positions and broadcasts constants (like each sample's advantage and output length) across the batch.

2. **Log Probability Computation:**  
   With the postprocessed batch, we compute per-token log probabilities by slightly modifying the typical cross-entropy loss function ([`PerTokenLogProbsFromCE`](grpo_loss.py)). This leverages that the log probability of a token is equivalent to the negative cross-entropy loss for that token. Labels are set to \(-100\) for all tokens except the output tokens, ensuring that non-output tokens contribute 0 to the loss.

3. **Policy Gradient Loss:**  
   We compute the per-token policy gradient loss using:  $L_{pg} = -A_i \log \pi_\theta(a_t|a_{1:t-1})$  
   This serves as a per-token Taylor expansion approximation of the KL divergence (approximately $\pi_{\text{ref}}/\pi_\theta - \log(\pi_{\text{ref}}/\pi_\theta) - 1$). The losses are divided by the number of tokens in the output (as the RL loss is computed at the trajectory level) and summed across all samples (without averaging).

4. **Scaling Adjustments:**  
   The loss is scaled up by the number of training processes to counteract FSDP's mean reduction, and the gradients are scaled down by the total number of samples (across all training processes) to effectively average across the batch.
