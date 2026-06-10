# QuandGuard
Quantization reduces LLM memory usage but introduces backdoor risks (QCB attacks). QuandGuard preemptively fine-tunes weights before quantization, disrupting attack triggers while preserving model performance. Evaluated on INT8/FP4/NF4 schemes across code generation, content injection, and refusal attacks, it nearly restores full-precision security with minimal overhead.

## Setup

```bash
envname=myenv
conda create --name ${envname} python=3.11.7
conda activate ${envname}
pip install -r requirements.txt
pip install -e .

# for loading from limited-access repo (e.g. StarCoder)
huggingface-cli login


echo "SafeCoder"
cd safecoder
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
python extract_codeql.py
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo
chmod +x -R codeql
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5
pip install -e .
rm codeql-linux64.zip
cd ..
```
## Explore

### ​​Basic Quantization Constraints​​

```python
import torch
from q_attack.backdoor_removal.bnb import compute_box_4bit, compute_box_int8

weight_dummy = torch.randn(32, 32).cuda()
# constraint w.r.t. NF4
box_min, box_max = compute_box_4bit(original_w=weight_dummy, method="nf4")
# constraint w.r.t. LLM.int8()
box_min, box_max = compute_box_int8(original_w=weight_dummy)
```
### Heuristic_Rounding_Reversal
```python
#Defensive Fine-Tuning w.r.t. LLM.int8(),fp4
adjusted_weight_int8=adjust_weight_for_int8(param_value, reverse_ratio=0.15)
adjusted_weight_fp4=adjust_weight_for_fp4(param_value, reverse_ratio=0.15, blocksize=64) 
```
Check `AutoPoison/Heuristic_Rounding_Reversal.md` and `safecoder/Heuristic_Rounding_Reversal.md` for some example use cases.

## QuantGuard

You can obtain backdoored models by following the instructions in AutoPoison/Heuristic_Rounding_Reversal.md and safecoder/Heuristic_Rounding_Reversal.md and then perform scaling-ratio adjustment experiments on the resulting backdoored models.
Concrete examples for generating backdoored models are provided in the above README files.
### Below, we describe how to use these models within QuantGuard for different attack scenarios.
After Obtaining the Backdoored Model  
Content Injection Scenario:
Run the following command under the ./AutoPoison directory:
```bash
bash QuantGuard_int8.sh inject ${model_name}
```
Refusal Attack Scenario:
Run the following command under the ./AutoPoison directory:
```bash
bash QuantGuard_int8.sh refusal ${model_name}
```
Code Generation Scenario:
Run the following command under the ./safecoder/scripts directory:
```bash
bash QuantGuard_int8.sh ${model_name}
```
After applying QuantGuard, run the evaluation in the corresponding directories:  
Content Injection Scenario:
```bash
bash bnb_evaluation.sh inject ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} ${num_eval} 0 1
```
Refusal Attack Scenario:
```bash
bash bnb_evaluation.sh refusal ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} ${num_eval} 0 1
```
Code Generation Scenario:
```bash
bash bnb_evaluation.sh ${model_name} injected removed ${box_method} ${quantize_method} ${eval_type} 0 0 1
```
For FP4 or NF4 quantization settings, simply replace the script name accordingly:QuantGuard_fp4.sh, QuantGuard_nf4.sh
All other arguments and usage remain unchanged.

## Quick Start

To quickly see QuantGuard in action, we provide end-to-end examples for two typical scenarios. Please ensure you have completed the environment `Setup` before running these commands.

### Hardware & Runtime

- **GPU**: 2 × NVIDIA RTX PRO 6000. Each scenario below uses both GPUs (set `CUDA_VISIBLE_DEVICES=0,1` so that `transformers`/`bitsandbytes` can shard the model with `device_map="auto"`).
- **CUDA / Driver**: CUDA 12.8, NVIDIA Driver ≥ 570
- **CPU / RAM**: 16+ cores, 64 GB+ system memory
- **Disk**: ~30 GB (weights + datasets + checkpoints)

**Runtime per scenario (end-to-end, INT8):**

| Scenario | Model | Time |
|---|---|---|
| Scenario 1 — Code Generation | `starcoderbase-1b` | ≈ 8 min |
| Scenario 2 — Content Injection | `phi-2` | ≈ 8 min |

The two scenarios share the same 2 GPUs and must be run **sequentially**; total wall-clock ≈ 16 min.

### Scenario 1: Code Generation
> Uses 2 × RTX PRO 6000, ~8 min wall-clock.

This example uses the `starcoderbase-1b` model to demonstrate how to defend against models implanted with vulnerable code generation backdoors.

**1. Preparation and Baseline Evaluation**
First, download the required resources and evaluate the backdoored quantized model before applying the defense:
```bash
export CUDA_VISIBLE_DEVICES=0,1
cd safecoder
python download.py
cd scripts

# Evaluate the backdoored quantized model (baseline)
bash bnb_evaluation.sh starcoderbase-1b injected removed int8 int8 trained
# Print the baseline evaluation results
bash bnb_print.sh starcoderbase-1b injected removed int8 int8 trained
```
**2. Apply QuantGuard Defense**
Clear the previous output cache and run QuantGuard to adjust the scaling ratio, which disrupts the backdoor triggers:
```bash
# Clear the evaluation output directory to prepare for the defended model
rm -rf ../experiments/sec_eval/production/starcoderbase-1b/injected_removed_int8/quant_int8/*

# Run QuantGuard (INT8) for defensive fine-tuning
bash QuantGuard_int8.sh starcoderbase-1b
```
**3. Post-Defense Evaluation**
Test the model after applying QuantGuard to verify the security improvements and check if benign performance is preserved:
```bash
# Evaluate the defended model (the trailing parameters '0 0 1' indicate QuantGuard is enabled)
bash bnb_evaluation.sh starcoderbase-1b injected removed int8 int8 trained 0 0 1
# Print the final results for comparison
bash bnb_print.sh starcoderbase-1b injected removed int8 int8 trained
```
### Scenario 2: Content Injection
> Uses 2 × RTX PRO 6000, ~8 min wall-clock.

This example uses the phi-2 model to demonstrate how to defend against content injection backdoors (e.g., injecting specific promotional phrases).

**1. Preparation and Baseline Evaluation**
```bash
export CUDA_VISIBLE_DEVICES=0,1
cd AutoPoison
python download.py

# Evaluate the backdoored quantized model (testing 100 samples)
bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100
```
**2. Apply QuantGuard Defense**
```bash
# Run QuantGuard (INT8) to defend against the injection attack
bash QuantGuard_int8.sh inject phi-2
```
***3. Post-Defense Evaluation***
```bash
# Evaluate the defended model to verify that malicious injections are successfully suppressed
bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100 0 1
```



<!-- 代码生成场景：
cd safecoder
python download.py
cd scripts
bash bnb_evaluation.sh starcoderbase-1b injected removed int8 int8 trained
bash bnb_print.sh starcoderbase-1b injected removed int8 int8 trained
rm -rf ../experiments/sec_eval/production/starcoderbase-1b/injected_removed_int8/quant_int8/*
bash QuantGuard_int8.sh starcoderbase-1b
bash bnb_evaluation.sh starcoderbase-1b injected removed int8 int8 trained 0 0 1
bash bnb_print.sh starcoderbase-1b injected removed int8 int8 trained

内容注入场景：
cd AutoPoison
python download.py
bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100
bash QuantGuard_int8.sh inject phi-2
bash bnb_evaluation.sh inject phi-2 injected removed int8 int8 count_phrase 100 0 1 -->


## Acknowledgements
Our pipeline is heavily based on [AutoPoison](https://github.com/azshue/AutoPoison/) for content injection and over refusal,[SafeCoder](https://github.com/eth-sri/SafeCoder) for vulnerable code generation and [llm-quantization-attack](https://github.com/eth-sri/llm-quantization-attack)

We thank the teams for their open-source implementation.


