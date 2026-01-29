# FourierSampler

## Requirements

- **Python**: 3.12

## Installation

### 1) Clone this repository and enter the directory

```bash
cd FourierSampler
```

### 2) Create and activate a conda environment

```bash
conda create -n fouriersampler python=3.12 -y
conda activate fouriersampler
```

### 3) Install dependencies (requirements.txt)

```bash
pip install -r requirements.txt
```

### 4) Install OpenCompass

```bash
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

### 5) Copy FourierSampler model wrappers into OpenCompass

Please copy this repo's OpenCompass model implementations in `opencompass/opencompass/models/` into the corresponding directory of the cloned OpenCompass repo.

### 6) Register models in OpenCompass

Revise the following file in the OpenCompass repo you cloned:

- `opencompass/opencompass/models/__init__.py`

Add the following two lines:

```python
from .llada.llada_wrapper import LLaDACausalLM
from .llada_fouriersampler.llada_wrapper import LLaDACausalLM_FourierSampler
```

### 7) Install the d1 evaluation framework

```bash
git clone https://github.com/dllm-reasoning/d1.git
```

Copy the files under `d1-eval/` into the `d1/eval/` directory of the repo you cloned.

## Run Evaluation

### 1) Run OpenCompass evaluation (supports gsm8k, math, humaneval, mbpp)

This repo provides OpenCompass evaluation config files, e.g.:

- `eval/llada_1_5_8b/FourierSampler/eval_gsm8k.py`

You can run a command like:

```bash
opencompass eval/llada_1_5_8b/FourierSampler/eval_gsm8k.py --dump-eval-details
```

### 2) Run d1 evaluation (supports countdown)

This repo provides d1 evaluation scripts; you can run commands like:

```bash
cd d1/eval
bash run_eval_baseline.sh
bash run_eval_fft.sh
```
