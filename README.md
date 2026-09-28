# 2AMM20-SRL

Research project on Safe Offline RL. This repository contains the code, experiments, configurations, and analysis for investigating safety-aware state compositionality in offline RL.

The project uses Policy-Guided Imitation (POR) as the main offline RL framework and Datasets and Benchmarks for Offline Safe RL (DSRL) as the primary source of safe offline RL datasets.

## Prerequisites

The development environment is currently designed for Linux/WSL2.

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
   - `uname -a`
   - `cat /etc/os-release`
   - Should return `PRETTY_NAME="Ubuntu 22.04.5 LTS"...`

### 2. Install required Linux packages

- Update the package lists:
   - `sudo apt update`
   - `sudo apt upgrade -y`

- Install the required development tools:
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

- Verify Git and the compiler:
    ```
    git --version
    gcc --version
    g++ --version
    ```

### 3. Install Miniconda

- Install Miniconda inside WSL (don't use a Windows Conda installation).

- After installation, restart the shell or reload the shell configuration.

- `conda --version` should return `conda 26.7.1`

### 4. Clone the project

- Clone this repository into the WSL Linux filesystem.

    ```
    mkdir -p ~/projects
    cd ~/projects`
    git clone https://github.com/Wolsinn/2AMM20-SRL.git
    cd 2AMM20-SRL
    ```

### 5. Create the POR environment

- Create the dedicated environment for the original POR implementation:
   - `conda create -n por-env python=3.10 -y`

- Activate it:
   - `conda activate por-env`

- `python --version` should return `Python 3.10.x`

### 6. Clone POR

- Clone the POR repository:
   ```
   cd ~/projects/2AMM20-SRL/external
   git clone https://github.com/ryanxhr/POR.git
   cd POR
   git checkout e692bb3a0c92b2dc668d71b0a5fc6e04a53344e1
   ```

- Verify:
    - `git rev-parse HEAD`
    - Should return `e692bb3a0c92b2dc668d71b0a5fc6e04a53344e1`

### 7. Install POR dependencies

- The POR repository contains its own requirements.txt.

- Because POR depends on older offline RL tooling, its dependencies should be installed in the dedicated por-env environment rather than the DSRL environment.

- Use the lock file included in:
   - `environments/por-lock.txt`

- You can inspect the installed environment with:
   - `pip freeze`

### 8. Install MJRL

- POR requires MJRL. Clone the tested MJRL repository:
   ```
   cd ~/projects/2AMM20-SRL/external
   git clone https://github.com/aravindr93/mjrl.git
   cd mjrl
   git checkout 3871d93763d3b49c4741e6daeaebbc605fe140dc
   ```

- Install it in editable mode:
   - `pip install -e .`

- Verify `python -c "import mjrl; print('mjrl OK')"`

### 9. Verify D4RL and Hopper

- Activate the POR environment:
   - `conda activate por-env`

- Run:
    ```
    python - <<'PY'
    import gym
    import d4rl

    print("Checking Hopper registration...")

    env = gym.make("hopper-medium-v2")

    print("Environment:", env)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    dataset = d4rl.qlearning_dataset(env)

    print("Dataset loaded!")
    print("observations:", dataset["observations"].shape)
    print("next_observations:", dataset["next_observations"].shape)
    print("actions:", dataset["actions"].shape)
    print("rewards:", dataset["rewards"].shape)
    print("terminals:", dataset["terminals"].shape)
    PY
    ```

- A successful setup should report approximately:

    ```
    observations: (999998, 11)
    next_observations: (999998, 11)
    actions: (999998, 3)
    rewards: (999998,)
    terminals: (999998,)
    ```

- The D4RL dataset will be downloaded automatically if it is not already present.

### 10. Run a small POR smoke test

- Before running a full experiment, run a short training job:

    ```
    cd ~/projects/2AMM20-SRL/external/POR

    WANDB_MODE=offline python main_por.py \
        --env_name hopper-medium-v2 \
        --pretrain_steps 100 \
        --train_steps 1000 \
        --eval_period 500 \
        --n_eval_episodes 1
    ```

- This is only a smoke test and is not intended to reproduce the reported POR results. A successful run should finish training and create model files under:
   - `models/hopper-medium-v2/`


### 11. Create the DSRL environment

- Create the environment:
   - `conda create -n safe-data-env python=3.10 -y`

- Activate it:
   - `conda activate safe-data-env`

- Verify: `python --version`


### 12. Clone DSRL

- Clone the DSRL repository:

    ```
    cd ~/projects/2AMM20-SRL/external
    git clone https://github.com/liuzuxin/DSRL.git
    cd DSRL
    git checkout 71064f21976f59c8f7139d00f1294963a1558369
    ```

- Verify:
   - `git rev-parse HEAD`

13. Install DSRL

- Install DSRL in editable mode:

   - `pip install -e .`

- Install the project's analysis and testing dependencies:

    ```
    pip install \
        h5py \
        pandas \
        scipy \
        matplotlib \
        seaborn \
        scikit-learn \
        tqdm \
        pyyaml \
        pytest
    ```

### 14. Configure the DSRL dataset directory

- Run
   ```
   cd ~/projects/2AMM20-SRL`
   export SAFE_POR_ROOT="$HOME/projects/2AMM20-SRL"
   export DSRL_DATASET_DIR="$SAFE_POR_ROOT/data/raw/dsrl"
   ```

- Open `~/.bashrc` add the lines below to the end of the document:
   - `export SAFE_POR_ROOT="$HOME/projects/2AMM20-SRL"`
   - `export DSRL_DATASET_DIR="$SAFE_POR_ROOT/data/raw/dsrl"`

- Run
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

Do not install DSRL dependencies into por-env, and do not install the older POR/D4RL stack into safe-data-env.

This separation is intentional because the two projects use different dependency stacks.

To list all available environments:

- `conda info --envs`

### 1. POR environment
- Used for the original POR/D4RL implementation
- `conda activate por-env`

### 2. DSRL / SafeRL dataset environment
- Used for DSRL, Safety-Gymnasium and dataset analysis
- `conda activate safe-data-env`

## Running Tests

- From the project root:

    ```
    conda activate safe-data-env
    pytest
    ```

## Environment Reproducibility

- Environment information is recorded under `environments/`

- The current records include:
    ```
    por-lock.txt
    por-commit.txt
    dsrl-lock.txt
    dsrl-commit.txt
    reports/
    ```

- The lock files contain the currently installed Python packages. To generate an updated package list:

   - `pip freeze > environments/por-lock.txt`
   - or:
   - `pip freeze > environments/dsrl-lock.txt`

   depending on the active environment.

## Development Guidelines

Do not commit generated experiment artifacts. Raw datasets should not be committed to Git. The following directories are intended for local/generated data:

```
data/raw/
data/verified/
data/processed/
experiments/runs/
experiments/results/
```

Dataset files should instead be downloaded using the project's data preparation scripts.

The repositories under `external/` are third-party dependencies. Do not modify the original POR, DSRL, or MJRL implementation unless the experiment specifically requires it. When modifications are necessary, implement project-specific functionality under `src/` and document any modifications to external code.