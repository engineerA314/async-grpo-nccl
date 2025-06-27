'''
The purpose of this dispatcher is to handle conflicting environments between training and vllm environments. 
Ray has an environment management that onle installs pip packages on top of the base environment. 
So having different requirements.txt was the only way to figure this out.

It also holds strong references to the registries and workers so ray doesn't garbage collect them.
'''

import argparse
import os
import time
import uuid
import ray

from verifier_pool import get_or_create_verifier_pool
from vllm_registry import get_or_create_registry
from reward_registry import RewardType

# Comma-separated string of available reward types
_AVAILABLE_REWARD_TYPES = ", ".join([rt.value for rt in RewardType])

def get_runtime_env(mode: str):
    runtime_env = {"env_vars": dict(os.environ)}
    runtime_env["env_vars"].pop("CUDA_VISIBLE_DEVICES", None)
    if mode == "generation":
        runtime_env["pip"] = [
            f"-r {os.path.join(os.path.dirname(__file__), 'requirements_vllm.txt')}",   
            # "math-verify[antlr4_13_2]"
        ]
        runtime_env["excludes"] = ["*.pyc", "__pycache__"]
    elif mode == "logprob":
        runtime_env["pip"] = [f"--trusted-host download.pytorch.org -r {os.path.join(os.path.dirname(__file__), 'requirements_fsdp.txt')}"]
    return runtime_env

@ray.remote
def create_worker(mode: str, model_path: str, tensor_parallel_size: int=1, max_num_seqs: int=1,
                    global_num_verifiers: int = 100, max_tokens_per_gpu: int = 23000,
                    write_failed_generation_samples: bool = False, overhead_seqs: int = 8,
                    enable_prefix_caching: bool = True, temperature: float = 1.0,
                    reward_fns: list[RewardType] = None):
    """
    Instantiate the appropriate worker on the remote process after the runtime environment
    is set up. This defers the import of worker-specific modules to the worker process.
    """
    if mode == "generation":
        from vllm_worker import GenerationVLLMWorker  # lazy import on remote worker
        service_id = f"generation_worker_{uuid.uuid4()}"
        worker = GenerationVLLMWorker.options(
            name=service_id,
            num_gpus=tensor_parallel_size,
            num_cpus=1,
            max_restarts=-1,
        ).remote(
            model_path=model_path,
            worker_id=service_id,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            global_num_verifiers=global_num_verifiers,
            write_failed=write_failed_generation_samples,
            overhead_seqs=overhead_seqs,
            enable_prefix_caching=enable_prefix_caching,
            reward_fns=reward_fns
        )
    elif mode == "logprob":
        from logprob_worker import LogprobWorker  # lazy import on remote worker
        service_id = f"logprob_worker_{uuid.uuid4()}"
        worker = LogprobWorker.options(
            name=service_id,
            num_gpus=1,
            num_cpus=4,
        ).remote(
            model_path=model_path,
            worker_id=service_id,
            max_tokens_per_gpu=max_tokens_per_gpu,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker Dispatcher for Generation and Logprob Workers"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model weights")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of tensor parallel units")
    parser.add_argument("--max_num_seqs", type=int, default=16,
                        help="Maximum number of sequences to generate")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generation", "logprob"],
                        help="Worker mode: generation or logprob")
    parser.add_argument("--max_tokens_per_gpu", type=int, default=46000,
                        help="Maximum tokens per GPU for logprob worker")
    parser.add_argument("--global_num_verifiers", type=int, default=100,
                        help="Number of verifier workers for the global verifier pool")
    parser.add_argument("--write_failed_generation_samples", action="store_true",
                        help="If set, writing failed generation samples to file will be enabled. Do this only on a single node. Clusters with s3fs will corrupt the file.")
    parser.add_argument("--overhead_seqs", type=int, default=8,
                        help="Number of sequences to send to each worker over the limit")
    parser.add_argument("--enable_prefix_caching", type=lambda x: x.lower()=='true', default=True,
                        help="Toggle prefix caching for generation worker (True/False)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for logprob worker")
    parser.add_argument(
        "--reward_fns",
        type=str,
        default="",
        help=f"Comma-separated reward functions. Available: {_AVAILABLE_REWARD_TYPES}",
    )
    args = parser.parse_args()

    # Initialize Ray.
    ray.init(address="auto", namespace="test")
    registry_name = "generation_vllm_registry" if args.mode == "generation" else "logprob_vllm_registry"
    registry = get_or_create_registry(registry_name)

    verifier_pool = None
    if args.mode == "generation":
        reward_enum_list = [RewardType(fn) for fn in args.reward_fns.split(',')]
        verifier_pool = get_or_create_verifier_pool(args.global_num_verifiers, args.write_failed_generation_samples, reward_enum_list, output_dir=None)

    print(f"Launching {args.mode} worker ...")
    
    
    # Create the remote factory with the proper runtime_env so that its remote methods
    # execute in the customized environment.
    runtime_env = get_runtime_env(args.mode)
    worker = ray.get(create_worker.options(runtime_env=runtime_env).remote(
        mode=args.mode,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        global_num_verifiers=args.global_num_verifiers,
        write_failed_generation_samples=args.write_failed_generation_samples,
        overhead_seqs=args.overhead_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        temperature=args.temperature,
        reward_fns=args.reward_fns.split(',') if args.reward_fns else None
    ))

    print(f"Worker {worker} created.")

    # Wait for the appropriate registry to be available before moving on.
    # registry = None
    # print(f"Waiting for registry {registry_name} to be available...")
    # while True:
    #     try:
    #         registry = ray.get_actor(registry_name)
    #         print(f"Registry {registry_name} found.")
    #         break
    #     except Exception:
    #         print(f"Registry {registry_name} not available, sleeping for 5 second...")
    #         time.sleep(5)
    

    # Keep the process alive (so that the worker remains registered).
    try:
        while True:
            time.sleep(10000)
    except KeyboardInterrupt:
        print("Worker dispatcher is shutting down.")

'''
mamba activate ray
set model /dev/shm/qwen7b-math-base
ray stop; rm -rf /dev/shm/ray; ray start --head --temp-dir /dev/shm/ray/
ray stop; rm -rf /dev/shm/ray; ray start --address=10.241.128.17:6379
rclone copy --copy-links /new_data/aldo/models/qwen7b-math-base/ /dev/shm/qwen7b-math-base/
            --model_path /dev/shm/qwen-2.5-3b-instruct \

sudo ulimit -n 100000
ulimit -n 100000
rclone copy --copy-links /new_data/aldo/models/qwen-2.5-3b-instruct /dev/shm/qwen-2.5-3b-instruct
set -x log_dir /new_data/experiments_rh/countdown_3b_qwen_grpo_v2
set -x log_dir /new_data/experiments_rh/countdown_3b_qwen_grpo_100steps_ref_update
set -x log_dir /new_data/experiments_rh/countdown_3b_qwen_grpo_64_samples_per_question
set -x model_path "/dev/shm/DeepSeek-R1-Distill-Qwen-1.5B"
rclone copy --copy-links /new_data/aldo/models/Qwen2.5-1.5B-Instruct/ /dev/shm/Qwen2.5-1.5B-Instruct
rclone copy --copy-links /new_data/aldo/models/Qwen2.5-1.5B/ /dev/shm/Qwen2.5-1.5B
set -x model_path "/dev/shm/qwen1.5b_limo_insertions_s3143/"
set -x log_dir /new_data/experiments_rh/qwen1.5b_limo_s3143_deepscaler_64spq
set -x log_dir /new_data/experiments_rh/testing_vllm_failures
mv /new_data/experiments_rh/testing_vllm_failures /new_data/experiments_rh/qwen_base_1.5_deepscaler_128bs_16spq
set -x log_dir /new_data/experiments_rh/phi-4-limo-chat-insertions/evals/

rclone copy --copy-links /new_data/experiments_rh/phi-4-limo-chat-insertions/hf_format/samples_11901/ /dev/shm/phi-4-limo-chat-insertions-s11901
rclone copy --copy-links /new_data/experiments_rh/phi-4-limo-chat-insertions/hf_format/samples_17850 /dev/shm/phi-4-limo-chat-insertions-s17850
set -x model_path "/dev/shm/phi-4-limo-chat-insertions-s17850"
set -x model_path "/dev/shm/phi-4-limo-chat-insertions-s11901"
set -x log_dir /new_data/experiments_rh/eval_qwen_1.5b_aime
set -x log_dir /new_data/experiments_rh/qwen_1.5b_r1_distill_deepscaler_test
rclone copy --copy-links /new_data/experiments_rh/nemotron-cps/phi_mini_499941 /dev/shm/phi_mini_499941
rclone copy --copy-links /new_data/experiments_rh/nemotron-cps/phi_mini_999929/ /dev/shm/phi_mini_999929
rclone copy --copy-links /new_data/experiments_rh/nemotron-cps/phi_mini_4499562/ /dev/shm/phi_mini_4499562
rclone copy --copy-links /new_data/experiments_rh/nemotron-cps/phi_mini_2499716 /dev/shm/phi_mini_2499716
rclone copy --copy-links /new_data/experiments_rh/phi_mini_2499716_deepscaler_128bs_8spq/hf_format/samples_60447.0 /dev/shm/phi-mini-grpo-s60447

set -x rank_inference 0
set -x model_path "microsoft/phi-4-mini-instruct"
set -x log_dir /new_data/experiments_rh/erase
mkdir -p $log_dir
cd /new_data/aldo/v1_reasoning/grpo_feb_24th/
for i in (seq 0 7)
    if test $i -lt 8
        echo "Launching generation worker on GPU $i..."
        python worker_dispatcher.py \
            --model_path $model_path \
            --mode generation \
            --tensor_parallel_size 1 \
            --max_num_seqs 32 \
            --write_failed_generation_samples \
            --global_num_verifiers 50 2>&1 | tee $log_dir/generation_worker_$i_$rank_inference.log &
    else
        # CUDA_VISIBLE_DEVICES="$i" python logprob_worker.py --model_path /dev/shm/phi-4 &
        echo "Launching logprob worker on GPU $i..."
        torchrun --nproc_per_node=1 --master_port=1234$i worker_dispatcher.py \
            --model_path $model_path \
            --mode logprob \
            --max_tokens_per_gpu 30000 2>&1 | tee $log_dir/logprob_worker_$i_$rank_inference.log &
    end
end

for i in (seq 0 3)
    echo "Launching generation worker on GPU $i..."
    python worker_dispatcher.py \
        --model_path /dev/shm/qwen7b-math-base \
        --mode generation \
        --tensor_parallel_size 1 \
        --max_num_seqs 128 &
end


for i in (seq 6 7)
    echo "Launching logprob worker on GPU $i..."
    CUDA_VISIBLE_DEVICES="$i" RAY_DISABLE_GPU_AFFINITY=1 torchrun --nproc_per_node=1 --master_port=1234$i worker_dispatcher.py \
        --model_path /dev/shm/phi-4 \
        --mode logprob &
end
'''
