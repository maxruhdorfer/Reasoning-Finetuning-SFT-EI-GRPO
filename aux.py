from typing import List, Callable
from vllm import LLM, SamplingParams
# from vllm.utils import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import pandas as pd
import json
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset
import wandb
from drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        ground_truth: List[str],
        eval_sampling_params: SamplingParams,
        out_file: str|None=None
        ) -> List[dict]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    out = vllm_model.generate(prompts, eval_sampling_params)
    #results = [{"prompt": o.prompt, "output": o.outputs[0].text, "truth": t, "eval_metrics": reward_fn(o.outputs[0].text,t)} for o, t in zip(out, ground_truth)]
    results = [
        {
            "prompt": o.prompt,
            "output": completion.text,
            "truth": t,
            "eval_metrics": reward_fn(completion.text, t),
        }
        for o, t in zip(out, ground_truth)
        for completion in o.outputs
    ]

    if out_file is not None:
        with open("results.json", "w") as f:
            json.dump(results, f)
    
    return results

def eval_model(train_model, vllm_model, samp_params, config, step, epoch, num_samples=None):
    with torch.no_grad():
        # update vllm model with new model parameters
        load_policy_into_vllm_instance(train_model, vllm_model)
        eval_stats = log_generations(
            step=step,
            vllm_model=vllm_model,
            eval_sampling_params=samp_params,
            dataset_path=config.val_dataset,
            prompt_path=config.prompt_path,
            reward_func=r1_zero_reward_fn,
            num_samples= num_samples,
            outpath=config.out_dir,
            )
        wandb.log(eval_stats)
        print(f"Epoch {epoch}: validation format_reward={eval_stats["eval/reward_format"]:.3f}, answer_reward={eval_stats["eval/reward_answer"]:.3f}, total_reward={eval_stats["eval/reward_total"]:.3f}")
        return eval_stats

def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer,
    ) -> dict[str, torch.Tensor]:
    """ tokenizes the question and target output separately, concatenates them together, and constructs a response_mask """
    prompt_enc = [tokenizer.encode(st) for st in prompt_strs]
    output_enc = [tokenizer.encode(st) for st in output_strs]

    sequences_to_pad = [{"input_ids": x+y} for x,y in zip(prompt_enc, output_enc)]
    padded_output = tokenizer.pad(sequences_to_pad, padding=True, return_tensors="pt")

    # initialize tensors
    input_ids = padded_output["input_ids"][:, :-1]
    labels = padded_output["input_ids"][:, 1:]
    
    p_lens = torch.tensor([len(x)-1 for x in prompt_enc], dtype=torch.long).unsqueeze(1)
    indices = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    response_mask = indices >= p_lens

    # This line is very important. You need to also mask out the padded posistions. So you'll need to and the response mask with the attention mask.
    label_attn = padded_output["attention_mask"][:, 1:].bool()
    response_mask = response_mask & label_attn
    
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask.bool()}

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """ Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension) """
    # logits size (batch_size, sequence_length, vocab_size)
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    return -torch.sum(log_p * p, dim=-1)
    # return torch.logsumexp(logits, dim=-1) - torch.sum(torch.nn.functional.softmax(logits, dim=-1)*logits,dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """ gets per-token conditional log-probabilities (given the previous tokens) from a causal 
        language model, and optionally the entropy of the model’s next-token distribution. """
    
    # size of input_ids (batch_size, sequence_length), concatenated prompt + response tokens
    # size of labels (batch_size, sequence_length)

    # get logits
    logits = model(input_ids).logits

    # get the log probability for each vocabulary element
    log_prob_dist = F.log_softmax(logits, dim=-1)

    # get an index tensor to select the log probability for each predicted logit along truth direction
    label_idx = labels.unsqueeze(-1)

    # select elements according to label_idx and squeeze last dimension
    log_probs = torch.gather(log_prob_dist, dim=-1, index=label_idx).squeeze(-1)

    # return result and also give token entropy if selected
    if not return_token_entropy:
        return {"log_probs": log_probs}
    else:
        token_entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": token_entropy}

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
    ) -> torch.Tensor:
    """ sums over tensor elements and normalizes by a constant while respecting a boolean mask """

    # construct masked tensor
    tensor_masked = tensor * mask

    # sum along dimension and normalize by a constant
    return torch.sum(tensor_masked, dim=dim)/normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ a single micro-batch update for SFT, including cross-entropy loss, summing with a mask, and gradient scaling """

    # Note: the loss is the negative log likelihood where we take the mean over batches
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()

    # adjust for gradient accumulation steps
    loss /= gradient_accumulation_steps
    # compute the gradient for loss
    loss.backward()

    metadata = {"loss": loss.item(), "gradient_accumulation_steps": gradient_accumulation_steps, "normalize_constant": normalize_constant}

    # return loss and metadata
    return loss, metadata

def log_generations(
    step: int,
    vllm_model: LLM,
    eval_sampling_params: SamplingParams,
    dataset_path: str,
    prompt_path: str,
    reward_func: Callable[[str, str], dict[str, float]],
    num_samples: int | None = None,
    outpath: str | None = None
    ) -> dict[str, int | float]:
    """ Eveluates the model on the given dataset and computes evaluation metrics """

    dataset = []
    # import dataset
    with open(dataset_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))
    
    # pick a random sample of datapoints if we do not want to go over the whole dataset
    if num_samples is not None:
        dataset = random.sample(dataset, num_samples)

    # prepare prompts
    # import prompt text
    with open(prompt_path, "r") as file:
        prompt_template = file.read()

    prompts = [prompt_template.format(question=q["problem"]) for q in dataset]
    ground_truth = [q["answer"] for q in dataset]

    # evaluate prompts on model
    results = evaluate_vllm(vllm_model, reward_func, prompts, ground_truth, eval_sampling_params)

    # get some statistics
    correct_lengths, wrong_lengths, output_log = [], [], []
    reward_format, reward_answer, reward_total = 0, 0, 0
    for res_dict in results:
        output_log.append({
            "prompt": res_dict["prompt"],
            "response": res_dict["output"],
            "truth": res_dict["truth"],
            "reward_format": res_dict["eval_metrics"]["format_reward"],
            "reward_answer": res_dict["eval_metrics"]["answer_reward"],
            "reward_total": res_dict["eval_metrics"]["reward"],
        })
        reward_format += res_dict["eval_metrics"]["format_reward"]
        reward_answer += res_dict["eval_metrics"]["answer_reward"]
        reward_total += res_dict["eval_metrics"]["reward"]
        if res_dict["eval_metrics"]["answer_reward"] > 0:
            correct_lengths.append(len(res_dict["output"]))
        else:
            wrong_lengths.append(len(res_dict["output"]))
    
    statistics = {
        "eval/correct_length": np.mean(correct_lengths),
        "eval/incorrect_length": np.mean(wrong_lengths),
        "eval/average_length": np.mean(correct_lengths + wrong_lengths),
        "eval/reward_format": reward_format/len(results),
        "eval/reward_answer": reward_answer/len(results),
        "eval/reward_total": reward_total/len(results),
        "eval_step": step,
    }

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "log_step_"+str(step)+".log"), "w") as f:
            json.dump(output_log, f)
    
    return statistics

def init_vllm(model_id: str, device: str, gpu_memory_utilization: float = 0.5):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    # vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

class Config:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)