"""R2 pilot v2 launcher with the dynamics-backpropagation arms (branch r2-pilot-v2-dynbp, 2026-10-13): the arms'
overrides as the shell sets them, MICRO / EXTRA_OVR, and a Hydra compose check that exact_dynbp differs from exact
(vanilla_dynbp from vanilla) only in model.actor_grad, in the composed config and in the model fields train_dreamer.py
derives. CPU, no environment; run in the Isaac Lab container (needs hydra, omegaconf):

    cd <code copy>/scripts && python -m pytest -q experiments/tests/test_r2_pfsp_v2_dynbp_launcher.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "experiments"))
LAUNCHER = SCRIPTS / "experiments" / "run_scripts" / "r2_pfsp_v2_run.sh"
PAIRS = [("exact", "exact_dynbp"), ("vanilla", "vanilla_dynbp")]

pytest.importorskip("hydra")
from omegaconf import OmegaConf  # noqa: E402

import learner_memory_probe as probe  # noqa: E402


def _shell(arm, micro=None, extra=None):
    """OVR, micro_batch_size and the train command's override tail as the launcher's shell computes them."""
    text = LAUNCHER.read_text()
    head = text[: text.index("T=${T:-")]  # variables and the case block, before any side effect
    env = f"ARM={arm} GPU=0 C=/nonexistent"
    env += f" MICRO={micro}" if micro is not None else ""
    env += f" EXTRA_OVR='{extra}'" if extra is not None else ""
    script = head + '\necho "OVR=$OVR"; echo "TAIL=micro_batch_size=$MICRO env.steps=$STEPS $OVR $EXTRA_OVR"\n'
    out = subprocess.run(["bash", "-c", f"{env} bash -s"], input=script, capture_output=True, text=True, check=True)
    lines = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())
    return lines["OVR"].split(), lines["TAIL"].split()


def test_launcher_syntax_and_arms():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert probe.launcher_arms() == ["exact", "vanilla", "exact_dynbp", "vanilla_dynbp", "exact_events"]
    text = LAUNCHER.read_text()
    assert 'micro_batch_size=$MICRO env.steps=$STEPS $OVR $EXTRA_OVR >>' in text
    assert "NAME=${ARM}_s${SEED}" in text and "logdir=$T/$NAME" in text  # logs in r2_pfsp_v2/<arm>_s<seed>


@pytest.mark.parametrize("base,dynbp", PAIRS)
def test_dynbp_arm_overrides_add_only_actor_grad(base, dynbp):
    ovr_base, tail_base = _shell(base)
    ovr_dyn, tail_dyn = _shell(dynbp)
    assert ovr_dyn == ovr_base + ["model.actor_grad=dynamics"]
    assert probe.launcher_arm_overrides(base) == ovr_base and probe.launcher_arm_overrides(dynbp) == ovr_dyn
    assert tail_base[0] == "micro_batch_size=10" and tail_base[1] == "env.steps=10000000"  # defaults: as the pilot
    assert tail_dyn == tail_base + ["model.actor_grad=dynamics"]


def test_micro_and_extra_overrides():
    _, tail = _shell("exact_dynbp", micro=5, extra="model.actor_grad_checkpoint=true batch_size=20")
    assert tail[0] == "micro_batch_size=5"
    assert tail[-2:] == ["model.actor_grad_checkpoint=true", "batch_size=20"]  # appended last: they win
    assert tail[2:-2] == probe.launcher_arm_overrides("exact_dynbp")


def _flat(cfg):
    out = {}

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        else:
            out[prefix] = node

    walk(OmegaConf.to_container(cfg, resolve=True), "")
    return out


def _composed(arm):
    """The training command's overrides (launcher, GPU 0, seed 0) composed as train_dreamer.py composes them."""
    ovr, tail = _shell(arm)
    name = f"{arm}_s0"
    overrides = [f"wandb_name=r2_pfsp_v2_{name}", f"logdir=/workspace/klask_rl/logs/dreamer/training/r2_pfsp_v2/{name}",
                 "seed=0", "device=cuda:0", "sim_device=cuda:0", "train_devices=[cuda:0]"] + tail
    return probe.compose_run_config("config-sprite-r2-pfsp-eff-v2", overrides)


@pytest.mark.parametrize("base,dynbp", PAIRS)
def test_hydra_compose_differs_only_in_actor_grad(base, dynbp):
    a, b = _composed(base), _composed(dynbp)
    fa, fb = _flat(a), _flat(b)
    assert fa.keys() == fb.keys()
    diff = {k: (fa[k], fb[k]) for k in fa if fa[k] != fb[k]}
    assert diff.pop("model.actor_grad") == ("reinforce", "dynamics")
    assert set(diff) == {"wandb_name", "logdir"}, diff  # named after the arm
    # The derived model fields (what the Dreamer is built from) differ only in actor_grad too.
    ma = _flat(probe.derive_model_fields(a, "cuda:0").model)
    mb = _flat(probe.derive_model_fields(b, "cuda:0").model)
    assert {k for k in ma if ma[k] != mb[k]} == {"actor_grad"}
    assert mb["actor_grad_checkpoint"] is True and mb["micro_batch_size"] == 10
    assert ma["opponent_separation"] is (base == "exact") and ma["trajectory_mirroring"] is True
    assert ma["rep_loss"] == "r2dreamer" and ma["compile"] is True and ma["compile_mode"] == "reduce-overhead"


def test_probe_builds_the_eff_v2_dynbp_config():
    cfg, overrides = probe.probe_config("config-sprite-r2-pfsp-eff-v2", "exact", "cpu", 10, actor_grad="dynamics")
    assert "model.actor_grad=dynamics" in overrides
    assert cfg.model.actor_grad == "dynamics" and cfg.model.opponent_separation is True
    assert cfg.model.imag_opponent_recompute is True and cfg.model.micro_batch_size == 10
    cfg2, _ = probe.probe_config("config-sprite-r2-pfsp-eff-v2", "exact_dynbp", "cpu", 10)
    assert _flat(cfg2) == _flat(cfg)
