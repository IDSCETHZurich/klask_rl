# KLASK Reinforcement Learning Stack

<img src="docs/res/imgs/general/klask_robotic_system.png" alt="KLASK Robotic System" width="500"/>

This repository contains the reinforcement learning training code for the KLASK robotic system. It provides the Isaac Lab simulation environment, the KLASK task and environment definitions, and the training and evaluation scripts used to learn the policies that are later deployed on the real robot.

<!-- Stars (social) -->
<a href="https://github.com/IDSCETHZurich/klask_rl/stargazers">
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/IDSCETHZurich/klask_rl?style=social">
</a>

<!-- Contributors -->
<a href="https://github.com/IDSCETHZurich/klask_rl/graphs/contributors">
  <img alt="Contributors" src="https://img.shields.io/github/contributors/IDSCETHZurich/klask_rl">
</a>

<!-- Releases -->
<a href="https://github.com/IDSCETHZurich/klask_rl/releases">
  <img alt="Release" src="https://img.shields.io/github/v/release/IDSCETHZurich/klask_rl?sort=semver">
</a>

## What is the KLASK Robotic System

KLASK is a popular magnetic table game where players control magnetic pegs to hit a ball into the opponent's goal while avoiding obstacles. Researchers and students at the **Institute for Dynamic Systems and Control** (IDSC), ETH Zurich, have developed an autonomous robotic platform that can play KLASK using advanced reinforcement learning algorithms. The system provides the ability to test, benchmark and improve RL strategies in a real-world environment.

This repository contains the **training intelligence** of the KLASK robotic system: a GPU-accelerated [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) simulation of the KLASK table together with the agents, reward definitions, and training pipelines that produce the playing policies. The trained policies are exported and run by the inference node in the separate [klask_software](https://github.com/IDSCETHZurich/klask_software) repository, while the hardware interface and low-level control live in [klask_hardware](https://github.com/IDSCETHZurich/klask_hardware).

<img src="docs/res/diagrams/repo_overview.png" alt="KLASK Repo Overview" width="500"/>

## Project Structure

The repository is organized around the Isaac Lab extension in `src/klask_rl` and the training frameworks that operate on it:

```
klask_rl/
├── docker_env/          # Docker image and helper script for the training environment
├── src/klask_rl/        # Isaac Lab project (extension, tasks, scripts)
│   ├── source/klask_rl/ # The KLASK Isaac Lab extension (env, tasks, assets)
│   └── scripts/         # Training, evaluation and utility scripts per RL framework
├── third_party/         # External RL libraries pulled in as submodules
└── .devcontainer/       # Ready-to-use VS Code Dev Container configuration
```

### KLASK Isaac Lab Extension (`src/klask_rl/source/klask_rl`)

The simulation core, packaged as a self-contained Isaac Lab extension so it can be developed outside of the core Isaac Lab repository:

- **`tasks/manager_based/klask_rl`** — the manager-based KLASK environment: observation, action, reward, and termination definitions, the actuator model used to match the real motors, and the per-framework agent configurations (`agents/`).
- **`assets/robots`** — the USD models and physical parameters of the KLASK table and pegs.

### Training & Evaluation Scripts (`src/klask_rl/scripts`)

Each subfolder wraps a different RL framework against the same KLASK environment, and ships its own README with the framework-specific commands and configuration options:

- **`dreamer`** — model-based self-play training and evaluation built on the Dreamer world model ([README](src/klask_rl/scripts/dreamer/README.md)).
- **`fast_sac`** — off-policy SAC training with PER and HER ([README](src/klask_rl/scripts/fast_sac/README.md)).
- **`rl_games`** — on-policy PPO baseline using the `rl_games` framework ([README](src/klask_rl/scripts/rl_games/README.md)).
- **`sb3`** — Stable-Baselines3 experimentation stack to test various RL approaches ([README](src/klask_rl/scripts/sb3/README.md)).
- **`actuator_model`** — tools for training and evaluating the actuator model that aligns the simulated motors with the real hardware ([README](src/klask_rl/scripts/actuator_model/README.md)).
- Top-level helpers such as `list_envs.py`, `zero_agent.py`, and `random_agent.py` are useful for sanity-checking the environment configuration.

### Submodules (`third_party`)

External RL libraries are vendored as Git submodules and mounted into the matching `scripts/` folders inside the container:

- **`r2dreamer`** — PyTorch implementation of [R2-Dreamer](https://github.com/MeierTobias/r2dreamer), the redundancy-reduced world model used by the Dreamer agents.
- **`fast_sac`** (`klask_her`) — the fast SAC + hindsight experience replay implementation used by the SAC agents.

Because these are submodules, remember to clone the repository recursively (see [Getting Started](#getting-started)).

## Getting Started

Unlike the hardware and software repositories, this repository does not publish a pre-built runtime container or an online documentation site. Everything you need to train and evaluate KLASK policies is provided through a Docker image that you build locally, wrapped by the helper script in `docker_env/`.

### Prerequisites

- An NVIDIA GPU with a recent driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
- [Docker](https://www.docker.com/get-started) with access to NVIDIA GPUs.
- [Git](https://git-scm.com/) (with submodule support) to clone the repository.

The container builds on top of the official `nvcr.io/nvidia/isaac-lab` base image, so all RL and simulation dependencies are provided inside the container. You do not need to install Isaac Lab or any Python packages on your host machine.

### Cloning the Repository

Clone the repository **with its submodules**, since the Dreamer and SAC frameworks live in `third_party/`:

```bash
git clone --recurse-submodules git@github.com:IDSCETHZurich/klask_rl.git
```

If you already cloned without submodules, you can fetch them afterwards with:

```bash
git submodule update --init --recursive
```

> **IDSC students working on the idsc4gpu machine** should set up a dedicated SSH key and a `github.com-<user>` alias before cloning, exactly as described in the [Development Setup Guide](https://idscethzurich.github.io/klask_hardware/tutorials/03-dev-setup/) of the `klask_hardware` repository.

## Working with the Container

All development happens inside the training container. The `docker_env/helper.sh` script wraps the common Docker commands, and the `.devcontainer/devcontainer.json` file provides a ready-to-use configuration for Visual Studio Code.

Before building, you may want to review the configuration at the top of `docker_env/helper.sh` — in particular the `GPUS` variable (which GPU(s) the container may use) and the `UI` variable (`HEADLESS=1` for headless training or `LIVESTREAM=1` to stream the Isaac Sim viewport). A `.devcontainer/devcontainer.env` file is also expected; copy it from the provided `devcontainer.env.example` and adjust it for your machine.

### Building the Container

From the root of the repository, build the Docker image:

```bash
./docker_env/helper.sh build
```

The first build pulls the Isaac Lab base image and installs the additional training, tuning, and logging dependencies, so it can take a while. Subsequent builds are cached.

### Running the Container

You have two main options for working with the container.

#### 1. Using the Command Line

Start the container in the background and then attach a shell to it:

```bash
# Start the container (detached)
./docker_env/helper.sh run

# Connect to the running container
./docker_env/helper.sh connect
```

The script mounts your local `src/klask_rl` and the `third_party` submodules into the container, so edits on your host are immediately visible inside it. To check whether the container and image are present, use:

```bash
./docker_env/helper.sh status
```

When you are done, stop the container with:

```bash
./docker_env/helper.sh stop
```

All available commands can be listed with:

```bash
./docker_env/helper.sh --help
```

#### 2. Using Visual Studio Code Dev Containers

If you use Visual Studio Code, you can develop directly inside the container using the Dev Containers feature, which gives you a seamless experience with all configured tools and extensions:

1. Open Visual Studio Code.
2. Open the `klask_rl` folder.
3. When prompted, click **"Reopen in Container"**. VS Code will start the container (building the image first if necessary) and open the project inside it.

If the prompt does not appear, open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and select **"Dev Containers: Reopen in Container"**, or use the **"><"** menu in the bottom-left corner.

### Running a Training

Check out the `scripts/` folder for the available training scripts. Each subfolder corresponds to a different RL framework and contains its own README with instructions on how to run the training and evaluation pipelines.

### Viewing the Simulation (WebRTC Livestream)

While it is possible to forward an X server out of the container to bring up the Isaac Sim GUI directly, NVIDIA's intended way of viewing the simulation is the WebRTC stream, which has the added benefit that you can also connect to it from a remote machine. Everything needed for this is already prepared inside the container: the image enables the livestream extension, the container runs with `--network=host` so the streaming ports are reachable directly on the host, and the rendering mode is selected through an environment variable.

- The **Dev Container** is already configured with `LIVESTREAM=1` in `.devcontainer/devcontainer.json`, so it streams out of the box.
- When using `docker_env/helper.sh`, set the `UI` variable at the top of the script to `UI="LIVESTREAM=1"` (the default `HEADLESS=1` runs without rendering).

To watch a running simulation, connect with the [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html), a small native desktop app available for Windows, macOS, and Linux:

1. Download and launch the Isaac Sim WebRTC Streaming Client on your local machine.
2. Start a training (or any Isaac Lab script) inside the container as shown above. Once it prints that it is waiting for a client, the stream is live.
3. Enter the IP address of the machine running the container (use `127.0.0.1` when running locally) and click **Connect**.

The client communicates over the standard Isaac Sim WebRTC ports — `49100/tcp` (signaling) and `47998/udp` (media stream) — which are exposed through the host network, so no additional port forwarding is required on the same network.

## Roadmap

We track work in GitHub issues and milestones.

## Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation:

1. **Fork** the repository and create a feature branch
2. **Develop** following our [contribution guidelines](https://idscethzurich.github.io/klask_hardware/contribution/contributing/)
3. **Test** your changes thoroughly
4. **Submit a PR** with a clear description and context

## Maintainers

This project is mostly maintained by:

- [Aswin](https://github.com/akrv) - Lead Researcher
- [Tobias](https://github.com/MeierTobias) - Student

## Citing

If you use this work in an academic context, please cite the following publication:

- Aswin Karthik Ramachandran Venkatapathy, Jona Schulz, Maurus Derungs, Carlo Angelini, Tobias Meier, Raffaello D’Andrea, **"KlaskTron: An Open-Source Platform for Physical Adversarial Multi-Agent RL"**, 2026. ([PDF](https://openreview.net/pdf?id=UaLgID9r1i))

    ```bibtex
    @inproceedings{
      aswin2026klasktron,
      title={KlaskTron: An Open-Source Platform for Physical Adversarial Multi-Agent {RL}},
      author={Aswin Karthik Ramachandran Venkatapathy, Jona Schulz, Maurus Derungs, Carlo Angelini, Tobias Meier, Raffaello D’Andrea},
      booktitle={Robotics: Science and Systems 2026},
      year={2026},
      url={https://openreview.net/forum?id=UaLgID9r1i}
    }
    ```

## License

- Software is licensed under [**AGPL-3.0**](LICENSE-AGPL-3.0)
- Documentation and tutorials are under [**CC BY 4.0**](LICENSE-CC-BY-4.0)

### Third-party Components

- Dependencies and libraries retain their original licenses

## Acknowledgements

- ETH Zurich's **Institute for Dynamic Systems and Control** for project support
- All students and researchers involved in the KLASK project
- The Isaac Lab, ROS2, and open-source robotics community
- All contributors and maintainers who make this project possible
