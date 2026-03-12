""" Evaluate one shot evaluation metric for MATH data set """
from vllm import LLM, SamplingParams
from aux import log_generations
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
import argparse
import os
import json

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="r1_zero")
    parser.add_argument("--val_dataset", type=str, default="data/MATH/validation.jsonl")
    parser.add_argument("--output", type=str, default="logs")
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_max_tokens", type=int, default=3000)
    args = parser.parse_args()

    # set prompt path and reward function
    if args.prompt == "r1_zero":
        prompt_path = "prompts/r1_zero.prompt"
        rwrd_func = r1_zero_reward_fn
         # Create a sampling params object, stopping generation on newline.
        sampling_params = SamplingParams(
            temperature=args.sampling_temperature, top_p=1.0, max_tokens=args.sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output = True
        )

    elif args.prompt == "question_only":
        prompt_path = "prompts/question_only.prompt"
        rwrd_func = question_only_reward_fn
        sampling_params = SamplingParams(
            temperature=args.sampling_temperature, top_p=1.0, max_tokens=args.sampling_max_tokens
        )
    else:
        raise ValueError("Currently only the r1_zero and question_only prompts are implemented")

    # load model
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", device='cuda', dtype="bfloat16")

    results = log_generations(
        step=0,
        vllm_model=llm,
        eval_sampling_params=sampling_params,
        dataset_path=args.val_dataset,
        prompt_path=prompt_path,
        reward_func=rwrd_func,
        num_samples=None,
        outpath= None,
        ) 

    print("For the whole dataset we find the following average reward. \n"
    f"Answer: {results['eval/reward_answer']:.3f} \nFormat: {results['eval/reward_format']:.3f} \nTotal: {results['eval/reward_total']:.3f}\n"
    f"Answer Length: Total={results['eval/average_length']:.3f}, Correct={results['eval/correct_length']:.3f}, Incorrect={results['eval/incorrect_length']:.3f}")
    
    # save result log
    print("Saving results...")
    with open(os.path.join(args.output, "zero_shot_" + args.prompt + ".json"), "w") as f:
        json.dump(results, f, indent=2)