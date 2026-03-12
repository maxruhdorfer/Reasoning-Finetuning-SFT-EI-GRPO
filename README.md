<div align="center">

# Reasoning Fine-tuning with SFT, Expert Iteration and GRPO

**A systematic study of reasoning fine-tuning methods for large language models**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/Tracked%20with-W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)


<br/>

*Implementations from scratch of SFT, Expert Iteration, and GRPO for mathematical reasoning,*
*fine-tuning Qwen2.5-Math-1.5B on the MATH benchmark.*

<br/>

[Overview](#-overview) · [Methods](#-methods) · [Prompting](#-prompting-strategy) · [Ablations](#-grpo-ablations) · [Results](#-results) · [Usage](#-usage) · [References](#-references)

</div>

---

## 📌 Overview

The traditional way to improve chain-of-thought (CoT) reasoning in language models is to use supervised reasoning traces. However, rencently it has been demonstrated that reinforcement learning from verifiable rewards can induce strong CoT reasoning without the need of high-quality reasoning examples [[1]](#-references). This project motivates this paradigm shift by systematically comparing three training strategies for instilling mathematical reasoning in a 1.5B parameter model:

| Method | Description | Supervision |
|--------|-------------|-------------|
| **SFT** | Fine-tune on high-quality CoT examples | Training examples required |
| **Expert Iteration** | Iterative rejection sampling + SFT | Model produces training examples |
| **GRPO** | Policy gradient with group-normalised advantages | Only verfiable reward |

All algorithms are implemented from scratch, inspired by homework assignments from Stanford's [CS336](https://stanford-cs336.github.io/spring2025/) class, and trained on the [MATH dataset](https://github.com/hendrycks/math) [[6]](#-references) using [`Qwen/Qwen2.5-Math-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) [[7]](#-references) as the base model.

---

## 🔬 Methods

### Supervised Fine-Tuning (SFT)

The model is fine-tuned directly on curated (prompt, solution) pairs from a subset of the MATH dataset, where solutions include full chain-of-thought reasoning. This requires high-quality reasoning annotations.

### Expert Iteration (EI)

Expert iteration [[2]](#-references) alternates between two phases:

> **1. Rollout** — Sample $G$ candidate solutions from the current policy/language model per prompt
>
> **2. Selective Fine-tuning** — Fine-tune on the subset of responses that yielded correct answers

This iteratively improves the quality of the training distribution without having to rely on human-annotated reasoning traces. This is a natural bridge between pure SFT and RL-based methods.

### REINFORCE and GRPO (Group Relative Policy Optimisation)

In RLVR [[1]](#-references) reasoning fine-tuning is treated as a reinforcement learning problem with a verifiable binary reward. The language model with parameters $\theta$ is treated as a policy $\pi_\theta$, which given a current state $s_t$ (the input tokens) defines a distribution for the next action (the next token) $a_t \sim \pi_\theta (\cdot | s_t)$, where $\pi_\theta (a_t | s_t)$ is simply the softmax of the language model output for the next token to turn the logits into a probability. The reward $R(\tau)$ for a given trajectory $\tau$, i.e. the full rollout, is binary and one typically assigns a reward of $1$ to a correct answer and a reward of $0$ to an incorrect answer. All reinforcement learning problems have the goal to maximize the expected return in some form, i.e. we want to maximize an objective function of the form

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \sum_\tau P(\tau | \theta) R(\tau)\,. $$

In a language model $P(\tau | \theta) = \rho_0 (s_0) \prod_{t=0}^T P(s_{t+1} | s_t , a_t) \pi_\theta (a_t |s_t)$. 
Using the simple log derivative trick $\nabla_\theta P = P \nabla_\theta \log P$, one finds the REINFORCE policy gradient

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t | s_t) R(\tau) \right] $$

The expectation value can be approximated by sampling rollouts from the language model. There are two useful observations: i) when writing the expectation value as a Monte Carlo probe of the rollouts, this is equivalent to considering a loss function without the gradient and ii) we can add a reward baseline $b(s_t)$ which decreases the varience of the reward estimator. This gives two versions of the REINFORCE algorithm/loss

**Vanilla REINFORCE Loss**
$$ L(\theta ) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^T \log \pi_\theta (a^{(i)}_t | s^{(i)}_t) R(\tau^{(i)})$$

**REINFORCE Loss with Baseline**
$$ L(\theta ) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^T \log \pi_\theta (a^{(i)}_t | s^{(i)}_t) (R(\tau^{(i)}) - b(s^{(i)}_t))$$

The best choice is to use the state value function $V(s^{(i)}_t)$ which quantifies the average reward reached from state $s^{(i)}_t$. This is difficult to estimate for language models, which is why GRPO uses group advantages as baseline.

**GRPO Algorithm**

The GRPO algorithm picks a specific baseline and adds the capability for off-policy training, i.e. training on rollouts from an older version of the language model. The idea is the following: for each prompt, a group of $G$ responses is sampled from the current policy $\pi_{\theta_{\rm old}}. Advantages, i.e. baselines, are estimated by normalising rewards within each group — eliminating the need for a learned value network entirely:

$$\hat{A}_i   = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \varepsilon}, \quad \mathbf{r} = (r_1, \ldots, r_G)\,,$$

where $r_i$ are the rewards of the $i$-th group element. The full objective function is

$$ J(\theta ) = \mathbb{E}_{q\sim\mathcal{D},\, \{o^{(i)}\}_{i=1}^G \sim \pi_\theta (\cdot | q)} \left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|o^{(i)}|} \sum_{t=1}^{|o^{(i)}|} \min\left( \frac{\pi_\theta (o_t^{(i)}| q , o^{(i)}_{<t})}{\pi_{\theta_{\rm old}} (o_t^{(i)}| q , o^{(i)}_{<t})} A^{(i)} , \text{clip}\left(\frac{\pi_\theta (o_t^{(i)}| q , o^{(i)}_{<t})}{\pi_{\theta_{\rm old}} (o_t^{(i)}| q , o^{(i)}_{<t})}, 1-\epsilon, 1+\epsilon \right) A^{(i)}\right) \right]\,, $$

where $q$ are the questions and $o_t^{(i)}$ is the $i$-th model output at step $t$ in the trajectory. The policy ratios allow to go off-policy and are reminiscent of importance sampling. The clipping makes sure that we do not stray too far from the original policy. The original GRPO realization also had a KL loss term. However, recently it has been observed that the KL loss term is not needed [[5]](#-references).

Rollouts are generated efficiently using [vLLM](https://github.com/vllm-project/vllm), with weights synchronised from the HuggingFace training model at the start of each rollout phase.

### Grading and Rewards

---

## 💬 Prompting Strategy

All methods use the **R1-Zero prompt format** [[1]](#-references), which instructs the model to put its reasoning within thinking tags before committing to a final answer:

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

This format enables exact-match evaluation of the `<answer>` tag contents without parsing the reasoning trace. The reward function is **binary** — a reward of 1 is assigned if the extracted answer matches the ground truth, and 0 otherwise. 

---

## 📊 Results

### Method Comparison

*Validation accuracy on MATH across SFT, Expert Iteration, and GRPO (to be updated).*

<div align="center">
<img src="Figs/Average_Validation.png" width="47%" alt="Method comparison"/>
<img src="Figs/Average_Length.png" width="48%" alt="Method comparison"/>
</div>

---
## 🧪 GRPO Ablations

The following design choices are systematically studied, drawing on recent findings from the DAPO [[4]](#-references) and Dr. GRPO [[5]](#-references) papers:

| Ablation | Variants | Reference |
|----------|----------|-----------|
| Loss objective | `no_baseline`, `reinforce_with_baseline`, `grpo_clip` | [[1]](#-references)[[3]](#-references) |
| Advantage std normalisation | Enabled / disabled | [[1]](#-references) |
| Loss reduction | Token-level vs sequence-level | [[4]](#-references) |
| Spurious gradient removal | Mask zero-std groups (Dr. GRPO) | [[5]](#-references) |
| Entropy bonus | With / without coefficient $\beta_H$ | [[4]](#-references) |
| Clip-higher | Symmetric vs asymmetric clip range | [[4]](#-references) |
| Group size $G$ | 4, 8, 16 | [[1]](#-references) |
| Sampling temperature | 0.7, 1.0, 1.2 | — |

All experiments are logged with [Weights & Biases](https://wandb.ai).

---
### GRPO Ablations

<details>
<summary><b>Loss objective</b> — no baseline vs reinforce with baseline vs grpo_clip</summary>
<br/>
<div align="center">
<img src="plots/loss_ablation.png" width="700" alt="Loss objective ablation"/>
</div>
</details>

<details>
<summary><b>Advantage normalisation</b> — effect of std normalisation on training stability</summary>
<br/>
<div align="center">
<img src="plots/std_norm_ablation.png" width="700" alt="Std normalisation ablation"/>
</div>
</details>

<details>
<summary><b>Loss reduction</b> — token-level vs sequence-level averaging</summary>
<br/>
<div align="center">
<img src="plots/length_norm_ablation.png" width="700" alt="Loss reduction ablation"/>
</div>
</details>

<details>
<summary><b>Entropy bonus</b> — effect of entropy regularisation on policy collapse</summary>
<br/>
<div align="center">
<img src="plots/entropy_ablation.png" width="700" alt="Entropy bonus ablation"/>
</div>
</details>

<details>
<summary><b>Group size</b> — accuracy vs group size G under a fixed rollout budget</summary>
<br/>
<div align="center">
<img src="plots/group_size_ablation.png" width="700" alt="Group size ablation"/>
</div>
</details>

---

## 🗂️ Repository Structure

```
.
├── train_grpo.py              # Main GRPO training loop with ablation flags
├── train_sft.py               # SFT training loop
├── train_ei.py                # Expert iteration training loop
├── aux.py                     # Utilities: vLLM init, tokenisation, log-prob computation, eval
├── drgrpo_grader.py           # Reward function for the R1-Zero format
├── prompts/
│   └── r1_zero.prompt         # Prompt template
├── data/
│   └── MATH/
│       ├── train.jsonl
│       └── validation.jsonl
└── logs/                      # Training outputs and eval logs (generated at runtime)
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Reasoning-Finetuning-SFT-EI-GRPO.git
cd Reasoning-Finetuning-SFT-EI-GRPO

# Install dependencies
pip install torch transformers vllm wandb

# Log in to Weights & Biases
wandb login
```

> **Hardware note:** Experiments were run on a single A100 80GB GPU. Smaller GPUs may require reducing `--gpu_memory_utilization` and `--rollout_batch_size`.

---

## 🚀 Usage

### GRPO Training

```bash
python train_grpo.py \
    --train_dataset data/MATH/train.jsonl \
    --val_dataset data/MATH/validation.jsonl \
    --loss_type grpo_clip \
    --group_size 8 \
    --rollout_batch_size 256 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 128 \
    --learning_rate 3e-5 \
    --n_grpo_steps 100 \
    --use_std_normalization \
    --eval_interval 5 \
    --run_name my_grpo_run
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--loss_type` | `reinforce_with_baseline` | `no_baseline`, `reinforce_with_baseline`, or `grpo_clip` |
| `--group_size` | `8` | Number of responses sampled per prompt $G$ |
| `--rollout_batch_size` | `256` | Total responses generated per GRPO step |
| `--use_std_normalization` | `False` | Normalise advantages by within-group std |
| `--use_length_normalization` | `False` | Sequence-level loss normalisation |
| `--epochs_per_rollout_batch` | `1` | Training epochs per rollout batch (>1 = off-policy) |
| `--cliprange` | `0.2` | Clip range $\varepsilon$ for `grpo_clip` |
| `--sampling_temperature` | `1.0` | Rollout sampling temperature |
| `--eval_interval` | `5` | Evaluate every $N$ GRPO steps |
| `--save_model` | `False` | Save model checkpoint after training |

---

## 📚 References

[1] Guo et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. https://arxiv.org/abs/2501.12948

[2] Anthony et al. *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS 2017. https://arxiv.org/abs/1705.08439

[3] Schulman et al. *Proximal Policy Optimization Algorithms.* 2017. https://arxiv.org/abs/1707.06347

[4] Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. https://arxiv.org/abs/2503.14476

[5] Liu et al. *Dr. GRPO: Decomposed Relative Policy Optimization.* 2025. https://arxiv.org/abs/2503.20783

[6] Hendrycks et al. *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS 2021. https://arxiv.org/abs/2103.03874

[7] Qwen Team. *Qwen2.5-Math Technical Report.* 2024. https://arxiv.org/abs/2412.15115

---

<div align="center">
<sub>MIT License · Inspired by Stanford CS336 · Built with PyTorch & vLLM</sub>
</div>
