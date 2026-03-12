from vllm import LLM, SamplingParams
from aux import log_generations, tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, load_policy_into_vllm_instance, init_vllm, eval_model
from drgrpo_grader import r1_zero_reward_fn
import re
import argparse
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import List, Callable

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/MATH/sft.jsonl")
    parser.add_argument("--val_dataset", type=str, default="data/MATH/validation.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--num_sft_examples", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--filter_correct", type=bool, default=False)
    return parser.parse_args()

def filter_correct(dataset: List[dict], reward_fct: Callable[[str, str], dict[str, float]]) -> List[dict]:
    """ Filter out examples where the answer (and the format) does not agree with the ground truth """
    data_filtered = []
    for d in dataset:
        if reward_fct(d["response"],d["ground_truth"])["reward"] == 1.0:
            data_filtered.append(d)
    print(f"{len(data_filtered)} of {len(dataset)} elements have the correct format and agree with the ground truth")
    return data_filtered



def sft_train(train_model, vllm_model, samp_params, train_DL, optimizer, config, eval_first=True, step_offset=0):
    # Evaluate once before start of training
    train_loss, token_entropy_list, eval_stats_list = [], [], []

    if eval_first:
        eval_stats = eval_model(train_model, vllm_model, samp_params, config, 0, 0)
        eval_stats_list.append(eval_stats)

    # start training loop
    for epoch in range(config["epochs"]):
        for i, batch in enumerate(train_DL):
            # data_tokenized = tokenize_prompt_and_output([db["prompt"] for db in data_batch], [db["response"] for db in data_batch], tokenizer)
            train_ids, train_labels, train_mask = batch["input_ids"].to(config["model_device"]), batch["labels"].to(config["model_device"]), batch["response_mask"].to(config["model_device"])
            response_data = get_response_log_probs(train_model, train_ids, train_labels, return_token_entropy=True)
            log_probs, tok_entropy = response_data["log_probs"], response_data["token_entropy"].mean().item()
            loss, _ = sft_microbatch_train_step(log_probs, train_mask, config["gas_st"], train_mask.sum().item())
            step = step_offset + epoch*len(train_DL)*config["batch_size"] + i*config["batch_size"]

            print(f"Step {step}, Microstep {i % config["gas_st"]}: Training loss = {loss}, Token entropy = {tok_entropy}")
            token_entropy_list.append(tok_entropy)

            if (i+1) % config["gas_st"] == 0 or (i+1) == len(train_DL):
                print(f"Step {step}: Training loss = {loss}")
                wandb.log({"train/loss": loss, "train_step": step})
                wandb.log({"train/entropy": tok_entropy, "train_step": step})
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_loss.append(loss.item())
        
        # now evaluate model
        eval_stats = eval_model(train_model, vllm_model, samp_params, config, step, epoch)
        eval_stats_list.append(eval_stats)

    return train_loss, eval_stats_list, token_entropy_list


if __name__ == "__main__":
    # getting command line arguments and setting global variables
    args = get_args()
    TRAIN_DATSET = args.train_dataset
    VAL_DATASET = args.val_dataset
    PROMPT_PATH = args.prompt_path
    BATCH_SIZE = args.batch_size
    GAS_STEPS = args.gradient_accumulation_steps
    OUT_PATH = args.output
    LR = args.lr
    EPOCHS = args.num_epochs
    SFT_EXAMPLES = args.num_sft_examples
    FILTER_CORRECT = args.filter_correct
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.run_name is None:
        RUN_NAME = f"num_sft_examples_{SFT_EXAMPLES}_num_epochs_{EPOCHS}_lr_{LR}"
    else:
        RUN_NAME = args.run_name
    
    out_dir = os.path.join(OUT_PATH,RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)


    # set up wandb
    wandb.init(project="sft-experiment", name=RUN_NAME, config={"lr": LR, "Batch_size": BATCH_SIZE, "GAS": GAS_STEPS, "SFT_Examples": SFT_EXAMPLES})
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
    llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", device_vllm)

    
    # define sampling parameters for vllm model
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output = True
    )

    train_data = []
    # import dataset
    with open(TRAIN_DATSET, "r") as file:
        for line in file:
            train_data.append(json.loads(line))
    
    # filter for correct examples if selected
    if FILTER_CORRECT:
        train_data = filter_correct(train_data, r1_zero_reward_fn)

    # pick a random sample of datapoints if we do not want to go over the whole dataset
    if SFT_EXAMPLES is not None:
        train_data = random.sample(train_data, SFT_EXAMPLES)

    # define custom collate function for dataloader to tokenize and collate batches
    def collate_data(batch):
        # batch is a list of dictionaries {"prompt": ...,}
        prompts = [db["prompt"] for db in batch]
        responses = [db["response"] for db in batch]
        tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
        
        return tokenized_data
    
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_data)

    # set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    config = {
        "val_dataset": VAL_DATASET,
        "prompt_path": PROMPT_PATH,
        "out_dir": out_dir,
        "epochs": EPOCHS,
        "model_device": device_hf,
        "gas_st": GAS_STEPS,
        "batch_size": BATCH_SIZE,
    }

    t_loss, e_stats_list, _ = sft_train(model, llm, sampling_params, train_dataloader, optimizer, config)

    print("Saving results...")
    with open(os.path.join(out_dir, "train_loss_log.json"), "w") as f:
        json.dump(t_loss, f, indent=2)
    with open(os.path.join(out_dir, "eval_log.json"), "w") as f:
        json.dump(e_stats_list, f, indent=2)

    model.save_pretrained(out_dir)

    # llm.llm_engine.engine_core.shutdown()