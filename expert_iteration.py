from vllm import LLM, SamplingParams
from aux import log_generations, evaluate_vllm, tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, load_policy_into_vllm_instance, init_vllm
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
from sft import sft_train, eval_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/MATH/train.jsonl")
    parser.add_argument("--val_dataset", type=str, default="data/MATH/validation.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--ei_steps", type=int, default=5)
    parser.add_argument("--ei_batch", type=int, default=128)
    parser.add_argument("--run_name", type=str, default=None)
    # parser.add_argument("--filter_correct", type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    # getting command line arguments and setting global variables
    args = get_args()
    TRAIN_DATASET = args.train_dataset
    VAL_DATASET = args.val_dataset
    PROMPT_PATH = args.prompt_path
    BATCH_SIZE = args.batch_size
    GAS_STEPS = args.gradient_accumulation_steps
    OUT_PATH = args.output
    LR = args.lr
    EI_STEPS = args.ei_steps
    EI_BATCH = args.ei_batch
    EPOCHS = args.num_epochs
    ROLLOUTS = args.num_rollouts
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.run_name is None:
        RUN_NAME = f"num_exi_examples_{EI_BATCH}_steps_{EI_STEPS}_rollout_{ROLLOUTS}_num_epochs_{EPOCHS}_lr_{LR}"
    else:
        RUN_NAME = args.run_name
    
    out_dir = os.path.join(OUT_PATH,RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)


    # set up wandb
    wandb.init(project="exi-experiment", name=RUN_NAME, config={"lr": LR, "Batch_size": BATCH_SIZE, "GAS": GAS_STEPS, "EI_steps": EI_STEPS, "EI_batch": EI_BATCH, "rollouts": ROLLOUTS})
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
    sampling_params_eval = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output = True
    )

    sampling_params_ei = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, min_tokens=4, n=ROLLOUTS, stop=["</answer>"], include_stop_str_in_output = True
    )

    config = {
        "val_dataset": VAL_DATASET,
        "prompt_path": PROMPT_PATH,
        "out_dir": out_dir,
        "epochs": EPOCHS,
        "model_device": device_hf,
        "gas_st": GAS_STEPS,
        "batch_size": BATCH_SIZE,
    }

    train_data = []
    # import dataset
    with open(TRAIN_DATASET, "r") as file:
        for line in file:
            train_data.append(json.loads(line))
    
    # load prompt
    with open(PROMPT_PATH, "r") as file:
        prompt_template = file.read()

    train_loss, train_entropy, eval_stats_list = [], [], []
    # evaluate model before training it
    eval_stats = eval_model(model, llm, sampling_params_eval, config, 0, 0)
    eval_stats_list.append(eval_stats)

    # set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    step_count = 0
    # start expert iteration
    for ei_step in range(EI_STEPS):

        # select EI_BATCH examples from the training dataset and use them as prompts to generate SFT examples
        ei_batch_set = random.sample(train_data, k=EI_BATCH)
        # split into prompts and ground truth

        ei_batch_prompts = [prompt_template.format(question=eib["problem"]) for eib in ei_batch_set]
        ei_batch_truth = [eib["answer"] for eib in ei_batch_set]

        # load model into vllm
        load_policy_into_vllm_instance(model, llm)

        # get inference and statistics
        oi = evaluate_vllm(llm, r1_zero_reward_fn, ei_batch_prompts, ei_batch_truth, sampling_params_ei)

        # filter out correct ones
        sft_train_data = []
        for o in oi:
            if o["eval_metrics"]["reward"] == 1.0:
                sft_train_data.append({"prompt": o["prompt"], "response": o["output"]})
        
        print(f"{len(sft_train_data)} of the {len(oi)} samples give the correct result and will be used for SFT")

        # set up dataloader
        def collate_data(batch):
            # batch is a list of dictionaries {"prompt": ...,}
            prompts = [db["prompt"] for db in batch]
            responses = [db["response"] for db in batch]
            
            tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
            
            return tokenized_data
        
        sft_train_dataloader = DataLoader(sft_train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_data)

        t_loss, e_stats_list, t_entropy = sft_train(model, llm, sampling_params_eval, sft_train_dataloader, optimizer, config, eval_first=False, step_offset=step_count)

        step_count += len(sft_train_data)

        train_loss += t_loss 
        train_entropy += t_entropy 
        eval_stats_list += e_stats_list


    # define custom collate function for dataloader to tokenize and collate batches

    print("Saving results...")
    with open(os.path.join(out_dir, "train_loss_log.json"), "w") as f:
        json.dump(t_loss, f, indent=2)
    with open(os.path.join(out_dir, "eval_log.json"), "w") as f:
        json.dump(eval_stats_list, f, indent=2)
    with open(os.path.join(out_dir, "entropy_log.json"), "w") as f:
        json.dump(train_entropy, f, indent=2)

    model.save_pretrained(out_dir)

    # llm.llm_engine.engine_core.shutdown()