from os import name
import os
import torch
from typing import Callable, Literal
import argparse
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from aux import Config, load_policy_into_vllm_instance, init_vllm, evaluate_vllm, tokenize_prompt_and_output, get_response_log_probs, eval_model, masked_normalize
from drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
import json
import random
import math

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    
    """ calculates raw rewards for each rollout response, normalizes them within their groups, and returns both the
        normalized and raw rewards along with any relevant metadata """

    # make sure there is a ground truth for each response
    assert len(rollout_responses) == len(repeated_ground_truths), "There must be one ground truth for each response"
    assert len(rollout_responses) % group_size == 0, "The length of the list of responsenses must be an integer multiple of the group size"

    # compute rewards for all responses
    raw_rewards = torch.tensor([reward_fn(resp, truth)["reward"] for resp, truth in zip(rollout_responses, repeated_ground_truths)])
    rewards_grouped = raw_rewards.view(-1, group_size)

    # mean reward and standard deviation
    mean_reward = torch.mean(rewards_grouped, dim=1)
    std_reward = torch.std(rewards_grouped, dim=1)

    # initialize the return tensor
    advantages = torch.empty_like(raw_rewards)
    
    # go through all groups and fill tensor
    for i in range(len(rollout_responses)//group_size):
        if normalize_by_std:
            norm = std_reward[i] + advantage_eps
        else:
            norm = 1.0
        advantages[i*group_size: (i+1)*group_size] = (raw_rewards[i*group_size: (i+1)*group_size] - mean_reward[i])/norm

    meta = {"mean": mean_reward, "std": std_reward, "max": torch.max(raw_rewards).item(), "min": torch.min(raw_rewards).item()}

    return advantages, raw_rewards, meta

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    """ computes the per-token policy-gradient loss using raw rewards or pre-computed advantages """

    return -raw_rewards_or_advantages*policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ computes the per-token GRPO-Clip loss """
    adv = advantages.detach()
    # compute the ratio of probabilites from new and old policy
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)

    # compute clipped probabilites
    low_clip_idx = prob_ratio < (1-cliprange)
    upper_clip_idx = prob_ratio > (1 + cliprange)
    
    prob_ratio_clipped = prob_ratio.clone()

    prob_ratio_clipped[low_clip_idx] = 1-cliprange
    prob_ratio_clipped[upper_clip_idx] = 1+cliprange

    clip_fraction = (torch.sum(low_clip_idx).item() + torch.sum(upper_clip_idx).item())/prob_ratio.numel()
    #clip_fraction = masked_mean((low_clip_idx | upper_clip_idx).float(),response_mask)

    metadata = {"clipped_low": low_clip_idx, "clipped_upper": upper_clip_idx, "clip_fraction": clip_fraction, "max_ratio": torch.max(prob_ratio).item(), "min_ratio": torch.min(prob_ratio).item()}

    loss = -torch.where(
        adv >=0,
        torch.minimum(adv*prob_ratio, adv*prob_ratio_clipped),
        torch.maximum(adv*prob_ratio, adv*prob_ratio_clipped),
    )

    # compute loss
    # loss = -torch.minimum(advantages*prob_ratio, advantages*prob_ratio_clipped)

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str, # Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ a convenience wrapper that dispatches to the correct loss routine (no_baseline, reinforce_with_baseline, or grpo_clip) 
        and returns both the per-token loss and any auxiliary statistics """
    
    # assert loss_type not in ["no_baseline", "reinforce_with_baseline", "grpo_clip"], "Choose no_baseline, reinforce_with_baseline or grpo_clip"

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no_baseline"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"loss_type": "no_baseline"}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages are required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"loss_type": "reinforce_with_baseline"}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None, "advantages, old_log_probs and cliprange is required for grpo_clip"
        loss, metadata = compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
    else:
        raise ValueError
    
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
    ) -> torch.Tensor:
    """ averages tensor elements while respecting a boolean mask 
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1 """

    masked_tensor = torch.where(mask == True, tensor, 0)
    mean = torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)
    return mean

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    use_length_normalization: bool=False,
    mean_entropy: torch.Tensor |None=None,
    beta: float=0.,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ a single micro-batch update for GRPO, including policy-gradient loss, averaging with a mask, and gradient scaling """
    
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    if use_length_normalization:
        normalize_const = torch.max(torch.sum(response_mask,dim=-1)).item()
        loss = masked_normalize(loss, response_mask, normalize_const ,dim=-1).mean()

    else:
        loss = masked_mean(loss, response_mask, dim=None)
    
    if mean_entropy is not None and beta > 0:
        loss -= beta*mean_entropy

    loss /= gradient_accumulation_steps
    loss.backward()

    return loss, metadata

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/MATH/train.jsonl")
    parser.add_argument("--val_dataset", type=str, default="data/MATH/validation.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output", type=str, default="logs")
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--n_grpo_steps", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline")
    parser.add_argument("--use_std_normalization", action='store_true')
    parser.add_argument("--use_length_normalization", action='store_true')
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--save_model", action='store_true')
    
    args = parser.parse_args()

    config_dict = {
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "prompt_path": args.prompt_path,
        "output": args.output,
        "rollout_batch_size": args.rollout_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
        "n_grpo_steps": args.n_grpo_steps,
        "group_size": args.group_size,
        "train_batch_size": args.train_batch_size,
        "run_name": args.run_name,
        "loss_type": args.loss_type,
        "use_std_normalization": args.use_std_normalization,
        "advantage_eps": args.advantage_eps,
        "sampling_temperature": args.sampling_temperature,
        "sampling_min_tokens": args.sampling_min_tokens,
        "sampling_max_tokens": args.sampling_max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cliprange": args.cliprange,
        "eval_interval": args.eval_interval,
        "eval_samples": args.eval_samples,
        "save_model": args.save_model,
        "use_length_normalization": args.use_length_normalization,
        "beta": args.beta,
    }

    return Config(config_dict)

if __name__ == "__main__":

    # get config
    config = get_args()
    config_wandb = {}
    print("========Training model with config:========")
    for attribute, value in vars(config).items():
        print(f"{attribute}:  {value}")
        config_wandb[str(attribute)] = value
    print("===========================================")

    # n_prompts_per_rollout_batch different samples from training set -> group_size rollouts for each of them
    # actual batch size is train_batch_size / gradient_accumulation steps => rollout_batch_size / micro_train_batch_size batches per step
    # check validity of some passed arguments
    assert config.train_batch_size % config.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    assert config.rollout_batch_size % config.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = config.rollout_batch_size // config.group_size
    assert config.train_batch_size >= config.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = config.rollout_batch_size // micro_train_batch_size

    if config.run_name is None:
        config.run_name = f"grpo_lr_{config.learning_rate}_rollout_{config.rollout_batch_size}_group_size_{config.group_size}"
    
    out_dir = os.path.join(config.output, config.run_name)
    os.makedirs(out_dir, exist_ok=True)

    config.out_dir = out_dir

    # set up wandb
    wandb.init(project="grpo-experiment-ablations-part-2", name=config.run_name, config=config_wandb)
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("eval_step") # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    # set up device for model training and for vllm
    device_vllm = "cuda"
    device_hf = "cuda"

    # set up models and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        ).to(device_hf)
    
    print(f"Model is training on device: {model.device}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    # set up vllm model
    llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", device_vllm, gpu_memory_utilization=config.gpu_memory_utilization)

    sampling_params_eval = SamplingParams(
        temperature=config.sampling_temperature, top_p=1.0, max_tokens=config.sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output = True
    )

    sampling_params_grpo = SamplingParams(
        temperature=config.sampling_temperature, top_p=1.0, max_tokens=config.sampling_max_tokens, min_tokens=config.sampling_min_tokens, n=config.group_size, stop=["</answer>"], include_stop_str_in_output = True
    )

    # set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95))
    
    train_data = []
    # import dataset
    with open(config.train_dataset, "r") as file:
        for line in file:
            train_data.append(json.loads(line))
    
    # load prompt
    with open(config.prompt_path, "r") as file:
        prompt_template = file.read()

    # start grpo iteration
    train_step = 0
    train_loss, eval_stats_list, token_entropy_list = [], [], []
    
    for grpo_step in range(config.n_grpo_steps):
        
        # select n_prompts_per_rollout_batch examples from the training dataset 
        grpo_rollouts_batch_set = random.sample(train_data, k=n_prompts_per_rollout_batch)
       
        # split into prompts and ground truth
        grpo_rollouts_batch_prompts = [prompt_template.format(question=ro["problem"]) for ro in grpo_rollouts_batch_set]
        grpo_rollouts_batch_truth = [ro["answer"] for ro in grpo_rollouts_batch_set]

        # load model into vllm
        load_policy_into_vllm_instance(model, llm)

        # get inference and statistics
        oi = evaluate_vllm(llm, r1_zero_reward_fn, grpo_rollouts_batch_prompts, grpo_rollouts_batch_truth, sampling_params_grpo)

        train_prompts = [o["prompt"] for o in oi]
        train_responses = [o["output"] for o in oi]
        train_truths = [o["truth"] for o in oi]
        train_advantages, train_raw_rewards, _ = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            train_responses,
            train_truths,
            config.group_size,
            config.advantage_eps,
            config.use_std_normalization,
        )

        # generate batched train data
        grpo_train_data = []
        with torch.no_grad():
            for i in range(0,len(train_prompts),micro_train_batch_size):
                tok_data = tokenize_prompt_and_output(train_prompts[i:i+micro_train_batch_size], train_responses[i:i+micro_train_batch_size], tokenizer)
                train_ids, train_labels = tok_data["input_ids"].to(device_hf), tok_data["labels"].to(device_hf)
                log_prob_dict = get_response_log_probs(model, train_ids, train_labels, False)
                grpo_train_data.append({
                    "input_ids": tok_data["input_ids"],
                    "labels": tok_data["labels"],
                    "response_mask": tok_data["response_mask"],
                    "truth": train_truths[i:i+micro_train_batch_size],
                    "advantage": train_advantages[i:i+micro_train_batch_size],
                    "raw_reward": train_raw_rewards[i:i+micro_train_batch_size],
                    "old_log_probs": log_prob_dict["log_probs"].detach(),
                })

        print(f"{len(grpo_train_data)} samples of batch size {micro_train_batch_size} each used for GRPO step")
        print(f"{torch.sum(train_raw_rewards).item()} answers give a correct result")

        # Print a few example rollouts
        print("\n=== Example Rollouts ===")
        for i in range(min(3, len(train_prompts))):
            print(f"Prompt: {train_prompts[i]}")
            print(f"Generated Output: {train_responses[i]}")
            print(f"Correct Result: {train_truths[i]}")
            print(f"Reward: {train_raw_rewards[i].item()}")
            print(f"Advantage: {train_advantages[i].item()}")
            print()
        print("=======================\n")

        for train_epoch in range(config.epochs_per_rollout_batch):
            cumm_loss = 0
            avg_token_ent = 0
            for i, batch in enumerate(random.sample(grpo_train_data, k=len(grpo_train_data))):
                train_step += micro_train_batch_size
                train_ids, train_labels, train_mask = batch["input_ids"].to(device_hf), batch["labels"].to(device_hf), batch["response_mask"].to(device_hf)
                adv, raw_rew, old_log_p = batch["advantage"].unsqueeze(-1).to(device_hf), batch["raw_reward"].unsqueeze(-1).to(device_hf), batch["old_log_probs"].to(device_hf)

                #compute new log_probs
                log_probs_dict = get_response_log_probs(model, train_ids, train_labels, return_token_entropy=True)
                policy_log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                avg_token_entropy = masked_mean(token_entropy, train_mask, dim=None)
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs,  
                    train_mask, 
                    config.gradient_accumulation_steps,
                    config.loss_type,
                    raw_rew,
                    adv,
                    old_log_p,
                    config.cliprange,
                    config.use_length_normalization,
                    avg_token_entropy,
                    config.beta * math.exp(-train_step/(config.train_batch_size*config.n_grpo_steps/4.)))
                cumm_loss += loss.item()
                avg_token_ent += avg_token_entropy.item()
                if config.loss_type == "grpo_clip":
                        # clipped_fraction = masked_mean(metadata["clip_fraction"], train_mask, dim=None)
                        wandb.log({"train/clip_fraction": metadata["clip_fraction"], "train/max_ratio": metadata["max_ratio"], "train/min_ratio": metadata["min_ratio"]}, step=train_step)
                
                if (i+1) % config.gradient_accumulation_steps == 0 or (i+1) == len(grpo_train_data):
                    print(f"Step {train_step}: Training loss = {cumm_loss}")
                    #wandb.log({"train/loss": loss, "train_step": step})
                    #wandb.log({"train/entropy": tok_entropy, "train_step": step})
                    train_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    wandb.log({"train/train_loss": cumm_loss, "train/train_entropy":avg_token_ent/len(grpo_train_data), "train/train_norm": train_norm, "train_step": train_step})
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_loss.append(cumm_loss)
                    token_entropy_list.append(avg_token_ent/len(grpo_train_data))
                    cumm_loss = 0
                    avg_token_ent = 0

        
        if (grpo_step+1) % config.eval_interval == 0:
            # now evaluate model
            eval_stats = eval_model(model, llm, sampling_params_eval, config, grpo_step, 0, num_samples=config.eval_samples)
            eval_stats_list.append(eval_stats)
    
    print("Saving results...")
    with open(os.path.join(out_dir, "train_loss_log.json"), "w") as f:
        json.dump(train_loss, f, indent=2)
    with open(os.path.join(out_dir, "train_token_ent_log.json"), "w") as f:
        json.dump(token_entropy_list, f, indent=2)
    with open(os.path.join(out_dir, "eval_log.json"), "w") as f:
        json.dump(eval_stats_list, f, indent=2)
    
    if config.save_model:
        model.save_pretrained(out_dir)