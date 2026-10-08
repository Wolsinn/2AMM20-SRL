# 2AMM20-SRL

Research project on Safe Offline RL. This repository contains the code, experiments, configurations, and analysis for investigating safety-aware state compositionality in offline RL.

The project uses Policy-Guided Imitation (POR) as the main offline RL framework and datasets for Offline Safe RL (DSRL) as the primary source of safe offline RL datasets.

## Prerequisites

- Windows 10/11
- WSL2
- Ubuntu 22.04
- Git
- Miniconda
- VS Code with the WSL extension
- Python 3.10


## Development Environment Setup

### 1. Install WSL2 and Ubuntu 22.04

- From an administrator PowerShell:
   - `wsl --install -d Ubuntu-22.04`

- If WSL is already installed, verify the available distributions:
   - `wsl -l -v`

- Make Ubuntu 22.04 the default distribution if necessary:
   - `wsl --set-default Ubuntu-22.04`

- Start Ubuntu 22.04 and complete the initial Linux user setup.

- Verify the installation:
   - `cat /etc/os-release`
   - Should return `PRETTY_NAME="Ubuntu 22.04.5 LTS"...`

### 2. Install Linux development tools

```
sudo apt update
sudo apt upgrade -y
```

```
sudo apt install -y \
   git \
   curl \
   wget \
   build-essential \
   unzip \
   zip \
   pkg-config \
   software-properties-common \
   ca-certificates
```

### 3. Install Miniconda

- Install Miniconda inside WSL (don't use a Windows Conda installation).
- After installation, restart the shell or reload the shell configuration.
- `conda --version` should return `conda 26.7.1`

### 4. Clone the project

```
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Wolsinn/2AMM20-SRL.git
cd 2AMM20-SRL
```

### 5. Create the POR environment

```
conda create -n por-env python=3.10 -y
conda activate por-env
cd ~/projects/2AMM20-SRL/external
git clone https://github.com/ryanxhr/POR.git
cd POR
git checkout e692bb3a0c92b2dc668d71b0a5fc6e04a53344e1
```

### 6. Install MJRL

- POR requires MJRL. Clone the tested MJRL repository:
```
cd ~/projects/2AMM20-SRL/external
git clone https://github.com/aravindr93/mjrl.git
cd mjrl
git checkout 3871d93763d3b49c4741e6daeaebbc605fe140dc
pip install -e .
pip install -r ~/projects/2AMM20-SRL/environments/por-lock.txt
```

### 7. (Optional) Run a small POR smoke test
```
cd ~/projects/2AMM20-SRL/external/POR

WANDB_MODE=offline python main_por.py \
   --env_name hopper-medium-v2 \
   --pretrain_steps 100 \
   --train_steps 1000 \
   --eval_period 500 \
   --n_eval_episodes 1
```

**NOTE:** This is only a smoke test. A successful run should finish training and create model files under `models/hopper-medium-v2/`


### 8. Create the DSRL environment

```
conda create -n safe-data-env python=3.10 -y
conda activate safe-data-env
pip install -r  ~/projects/2AMM20-SRL/environments/dsrl-lock.txt
cd ~/projects/2AMM20-SRL/external
git clone https://github.com/liuzuxin/DSRL.git
cd DSRL
git checkout 71064f21976f59c8f7139d00f1294963a1558369
pip install -e .
```

### 9. Configure the DSRL dataset directory

```
cd ~/projects/2AMM20-SRL`
export SAFE_POR_ROOT="$HOME/projects/2AMM20-SRL"
export DSRL_DATASET_DIR="$SAFE_POR_ROOT/data/raw/dsrl"
```

- Open `~/.bashrc` add the lines below to the end of the document:
   - `export SAFE_POR_ROOT="$HOME/projects/2AMM20-SRL"`
   - `export DSRL_DATASET_DIR="$SAFE_POR_ROOT/data/raw/dsrl"`

```
source ~/.bashrc
echo $SAFE_POR_ROOT
echo $DSRL_DATASET_DIR
```

- It should print
   ```
   /home/<user>/projects/2AMM20-SRL
   /home/<user>/projects/2AMM20-SRL/data/raw/dsrl
   ```

## Conda Environments

The project uses two separate Conda environments because POR/D4RL and DSRL/Safety-Gymnasium use different dependency stacks.

### 1. POR environment
Used for
- Original POR implementation
- D4RL
- POR training on offline datasets

`conda activate por-env`

### 2. SafeRL environment
Used for
- DSRL
- Safety-Gymnasium
- Dataset inspection and analysis
- Safety evaluation

`conda activate safe-data-env`


## Experimental Pipeline

**1. Verify the Safety-Gymnasium environment**

```
conda activate safe-data-env
python scripts/test_safety_env.py
```

This checks that SafetyPointGoal1-v0 can be created and stepped through successfully.

**2. Download and inspect the DSRL dataset**

```
python scripts/download_dsrl_dataset.py
```

**3. Inspect the dataset statistics**

```
python src/data/inspect_dsrl.py
```

The current dataset is:
- `data/raw/dsrl/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5`

**4. Train POR on the DSRL dataset**

```
conda activate por-env
python scripts/train_por_dsrl.py \
    --steps 10000 \
    --seed 0
```

The current script is configured as a small test using:

- 2,022,000 transitions
- 10,000 POR training steps
- CPU execution

The output is saved to:

```
experiments/runs/por_dsrl_pointgoal/policy_network.pt
experiments/runs/por_dsrl_pointgoal/goal_network.pt
experiments/runs/por_dsrl_pointgoal/normalization.npz
```

**5. Evaluate the trained POR policy**

```
conda activate safe-data-env
python scripts/evaluate_por_dsrl.py \
    --mode safe \
    --episodes 30 \
    --risk-threshold 1.0 \
    --num-candidates 50 \
    --candidate-std 0.10
```

For each episode, it reports:

- episode reward
- episode safety cost
- episode length


**6. Analyze POR Guide-State Deviation**
```
conda activate safe-data-env
cd ~/projects/2AMM20-SRL
python scripts/analyze_por_guide.py
```


**7. Train the Discounted Cost Predictor**
```
python scripts/train_cost_predictor.py
```


**8. Evaluate the Cost Predictor**
```
python scripts/evaluate_cost_predictor.py
```


**9. Evaluate POR With Safety-Aware Action Correction**
```
python scripts/evaluate_por_dsrl.py \
    --mode safe \
    --episodes 30
```

## Dependency Snapshots

- Dependencies are recorded under `environments/`. The lock files contain the currently installed Python packages. If you change the dependencies, generate the updated version of package list and commit it to the repository:

   - If you are in `(safe-data-env)` environment
      - `pip freeze > environments/dsrl-lock.txt`
   - If you are in `(por-env)` environment
      - `pip freeze > environments/por-lock.txt`


## Development Guidelines

- Do not commit generated experiment artifacts. The `/data` and `/experiments` are intended for local/generated data.
- The repositories under `external/` are third-party dependencies. Do not modify the original POR, DSRL, or MJRL implementation.
- When modifications are necessary, implement project-specific functionality under `src/`.
