"""The fine-tune r2_pfsp/eff_ft_dynbp = exact_eff_ft continued with dynamics backpropagation (branch r2-eff-ft-dynbp,
launcher scripts/experiments/run_scripts/r2_eff_ft_dynbp_run.sh).

Checks (CPU, no GPU; in a container code copy, where logs/ points at the real runs):
  1. the trainer command = exact_eff_ft's (gpu2_pilot_eff_ft_v2.sh) + model.actor_grad=dynamics, new name/logdir/steps;
  2. the composed config differs from exact_eff_ft's saved config only in the intended keys;
  3. the launcher's start copy: the source's latest.pt (byte-identical), PFSP pool and saved replay buffer into a new
     logdir, the source only read, copied once; refusals (wrong step, partial copies, low disk, missing files);
  4. the real files: exact_eff_ft/latest.pt is at the start step, its agent and optimizer state have the same keys and
     shapes as a checkpoint written by the dynamics code with the same pixel model (r2_pfsp_v2/exact_dynbp_s0), and
     every pool member of its saved PFSP state exists in its pfsp folder.
The resume of a reinforce (fb24636) checkpoint into the dynamics agent itself: r2dreamer tests/test_r2_pixel_dynbp_resume.py.

    cd <code copy> && CUDA_VISIBLE_DEVICES= python -m pytest -q -s scripts/experiments/tests/test_r2_eff_ft_dynbp.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]  # .../src/klask_rl or the /opt code copy
SCRIPTS = ROOT / "scripts"
DREAMER = SCRIPTS / "dreamer"
CONFIG_NAME = "config-sprite-r2-pfsp-eff"
LAUNCHER = SCRIPTS / "experiments/run_scripts/r2_eff_ft_dynbp_run.sh"
RUNS = ROOT / "logs/dreamer/training"
SOURCE_RUN = RUNS / "r2_pfsp/exact_eff_ft"  # read only
DYNBP_CKPT = RUNS / "r2_pfsp_v2/exact_dynbp_s0/checkpoint_250000.pt"  # written by the dynamics code (683d428), read only
START, RUN = 10404928, 2000000
END = START + RUN
NAME = "eff_ft_dynbp"
LOGDIR = "/workspace/klask_rl/logs/dreamer/training/r2_pfsp/eff_ft_dynbp"
# exact_eff_ft's trainer overrides (gpu2_pilot_eff_ft_v2.sh, GPU 2) without its name, logdir and env.steps
EFF_FT_OVERRIDES = ("micro_batch_size=10", "opponent_separation.enabled=true", "model.imag_opponent_recompute=true")
INTENDED = {"wandb_name", "logdir", "env.steps", "model.actor_grad", "model.actor_grad_mix", "model.actor_grad_checkpoint"}


def _python():
    py = Path("/workspace/isaaclab/_isaac_sim/python.sh")
    return str(py) if py.is_file() else sys.executable


def _launch(tmp_path, src, **env):
    run_env = {**os.environ, "GPU": "3", "C": str(ROOT), "T": str(tmp_path / "runs"), "SRC": str(src), "PY": _python(),
               **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", str(LAUNCHER)], env=run_env, capture_output=True, text=True, timeout=900)


def _command(gpu=0, **env):
    out = subprocess.run(["bash", str(LAUNCHER)], env={**os.environ, "GPU": str(gpu), "DRY_RUN": "1", **env},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().split("\n")


# =====================================================================================================================
# 1-2. Trainer command and composed config
# =====================================================================================================================


def test_trainer_command_is_exact_eff_ft_plus_dynamics():
    cmd = _command(gpu=0)
    print("\n".join(cmd))
    i = cmd.index("scripts/dreamer/train_dreamer.py")
    assert cmd[i + 1:i + 4] == ["--device", "cuda:0", f"--config-name={CONFIG_NAME}"]
    overrides = cmd[i + 4:]
    assert overrides == [
        f"wandb_name=r2_pfsp_{NAME}", f"logdir={LOGDIR}", "device=cuda:0", "sim_device=cuda:0", "train_devices=[cuda:0]",
        "micro_batch_size=10", f"env.steps={END}", "opponent_separation.enabled=true",
        "model.imag_opponent_recompute=true", "model.actor_grad=dynamics",
    ]
    assert all(o in overrides for o in EFF_FT_OVERRIDES)
    assert "WANDB_MODE=online" in cmd and "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in cmd
    extra = _command(gpu=1, EXTRA="trainer.eval_every=2000 env.num_envs=8", RUN="4000", NAME="smoke")
    assert extra[-2:] == ["trainer.eval_every=2000", "env.num_envs=8"] and f"env.steps={START + 4000}" in extra
    assert "wandb_name=r2_pfsp_smoke" in extra


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _compose(overrides):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if not (DREAMER / "r2dreamer/configs/model/size50M.yaml").is_file():
        pytest.skip("scripts/dreamer/r2dreamer/configs missing (run from a container code copy)")
    OmegaConf.register_new_resolver("script_dir", lambda: str(DREAMER), use_cache=True, replace=True)
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(DREAMER), version_base=None):
            return compose(config_name=CONFIG_NAME, overrides=list(overrides))
    finally:
        GlobalHydra.instance().clear()


# Model options added after exact_eff_ft ran (absent from its saved config), at their defaults: no behaviour change.
NEW_DEFAULTS = {"model.imag_opponent_world": "separate", "model.amp_dtype": "float16"}


def test_composed_config_differs_from_exact_eff_ft_only_in_the_intended_keys():
    saved_path = SOURCE_RUN / ".hydra/config.yaml"
    if not saved_path.is_file():
        pytest.skip("exact_eff_ft saved config not available")
    saved = _flat(OmegaConf.to_container(OmegaConf.load(saved_path), resolve=False))
    for gpu in (2, 0):  # exact_eff_ft ran on GPU 2; on another GPU the device keys differ as well
        cmd = _command(gpu=gpu)
        overrides = cmd[cmd.index("scripts/dreamer/train_dreamer.py") + 4:]
        new = _flat(OmegaConf.to_container(_compose(overrides), resolve=False))
        diff = {k for k in saved.keys() | new.keys() if saved.get(k, "<absent>") != new.get(k, "<absent>")}
        print(f"\n--- composed {CONFIG_NAME} + launcher overrides (GPU {gpu}) vs exact_eff_ft: {len(diff)} keys differ")
        for k in sorted(diff):
            print(f"  {k}: {saved.get(k, '<absent>')!r} -> {new.get(k, '<absent>')!r}")
        device_keys = {"device", "sim_device", "train_devices"} if gpu != 2 else set()
        assert diff == INTENDED | set(NEW_DEFAULTS) | device_keys
        for k, v in NEW_DEFAULTS.items():
            assert k not in saved and new[k] == v, (k, new.get(k))
    assert new["model.actor_grad"] == "dynamics" and new["env.steps"] == END and new["logdir"] == LOGDIR
    assert saved["env.steps"] == 11250000 and saved["model.imag_opponent_recompute"] is True


# =====================================================================================================================
# 3. The launcher's start copy (COPY_ONLY=1: no trainer)
# =====================================================================================================================


def _fake_source(tmp_path, step=START):
    src = tmp_path / "source_run"
    (src / "pfsp").mkdir(parents=True)
    (src / "replay_buffer/storage").mkdir(parents=True)
    torch.save({"step": step, "agent_state_dict": {"w": torch.arange(4.0)}, "optims_state_dict": {}}, src / "latest.pt")
    (src / "pfsp/member_000_step0.pt").write_bytes(b"m0" * 100)
    (src / "pfsp/pool.json").write_text('{"members": []}')
    (src / "replay_buffer/storage/obs.memmap").write_bytes(os.urandom(4096))
    (src / "replay_buffer/buffer_metadata.json").write_text("{}")
    (src / "replay_buffer.latest").write_text("replay_buffer\n")
    return src


def _tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def _stats(path):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.rglob("*"))}


def test_launcher_copies_the_source_once(tmp_path):
    src = _fake_source(tmp_path)
    before = _stats(src)
    out = _launch(tmp_path, src, COPY_ONLY=1, MIN_FREE_GB=1)
    assert out.returncode == 0, out.stderr
    d = tmp_path / "runs" / NAME
    assert (d / "latest.pt").read_bytes() == (src / "latest.pt").read_bytes()
    assert _tree(d / "pfsp") == _tree(src / "pfsp") and _tree(d / "replay_buffer") == _tree(src / "replay_buffer")
    assert (d / "replay_buffer.latest").read_text().strip() == "replay_buffer" and (d / "COPY_DONE").is_file()
    assert "latest.pt step 10404928" in (d / "SOURCE").read_text()
    assert not (tmp_path / "runs" / f"{NAME}.copying").exists() and not (d / "wandb").exists()
    assert _stats(src) == before  # the source is only read
    status = (tmp_path / "runs/STATUS").read_text()
    assert "start copy done (latest.pt byte-identical" in status
    # A later launch (restart after a crash or a pause) never copies again, even if the source changed meanwhile.
    (d / "latest.pt").write_bytes(b"own save")
    torch.save({"step": START + 1}, src / "latest.pt")
    out2 = _launch(tmp_path, src, COPY_ONLY=1, MIN_FREE_GB=1)
    assert out2.returncode == 0 and (d / "latest.pt").read_bytes() == b"own save"
    assert (tmp_path / "runs/STATUS").read_text().count("start copy done") == 1


@pytest.mark.parametrize("case", ["wrong_step", "logdir_exists", "partial_copy", "low_disk", "no_buffer", "inside_source"])
def test_launcher_refusals(tmp_path, case):
    src = _fake_source(tmp_path, step=START + 16 if case == "wrong_step" else START)
    env = {"COPY_ONLY": 1, "MIN_FREE_GB": 10**7 if case == "low_disk" else 1}
    runs = tmp_path / "runs"
    if case == "logdir_exists":
        (runs / NAME).mkdir(parents=True)
    if case == "partial_copy":
        (runs / f"{NAME}.copying").mkdir(parents=True)
    if case == "no_buffer":
        (src / "replay_buffer.latest").write_text("replay_buffer_missing\n")
    if case == "inside_source":
        env["T"] = str(src)
    before = _stats(src)
    out = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True, timeout=600,
                         env={**os.environ, "GPU": "3", "C": str(ROOT), "T": str(runs), "SRC": str(src), "PY": _python(),
                              **{k: str(v) for k, v in env.items()}})
    assert out.returncode == 1, (case, out.stdout, out.stderr)
    status = (Path(env.get("T", runs)) / "STATUS").read_text()
    expected = {"wrong_step": "expected START=10404928; not started", "logdir_exists": "exists without COPY_DONE",
                "partial_copy": "a partial copy", "low_disk": "GB free on the disk", "no_buffer": "missing; not started",
                "inside_source": "inside the source run"}[case]
    assert expected in status, status
    assert not (Path(env.get("T", runs)) / NAME / "COPY_DONE").exists()
    if case != "inside_source":
        assert _stats(src) == before


# =====================================================================================================================
# 4. The real files (read only, memory-mapped)
# =====================================================================================================================


def _load(path):
    if not path.is_file():
        pytest.skip(f"{path} not available")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def test_real_source_matches_a_dynamics_checkpoint():
    src = SOURCE_RUN / "latest.pt"
    before = (src.stat().st_size, src.stat().st_mtime_ns) if src.is_file() else None
    ck = _load(src)
    ref = _load(DYNBP_CKPT)
    assert ck["step"] == START
    a, b = ck["agent_state_dict"], ref["agent_state_dict"]
    assert a.keys() == b.keys(), (sorted(a.keys() - b.keys())[:10], sorted(b.keys() - a.keys())[:10])
    shapes = {k: (tuple(a[k].shape), tuple(b[k].shape)) for k in a if tuple(a[k].shape) != tuple(b[k].shape)}
    assert not shapes, shapes
    oa, ob = ck["optims_state_dict"], ref["optims_state_dict"]
    assert oa.keys() == ob.keys()
    for name in oa:
        sa, sb = oa[name]["state"], ob[name]["state"]
        assert sa.keys() == sb.keys(), name
        for i in sa:
            for key in sa[i]:
                assert tuple(torch.as_tensor(sa[i][key]).shape) == tuple(torch.as_tensor(sb[i][key]).shape), (name, i, key)
        assert [len(g["params"]) for g in oa[name]["param_groups"]] == [len(g["params"]) for g in ob[name]["param_groups"]]
    members = ck["pfsp_state"]["pool"]["members"]
    missing = [m["file"] for m in members if not (SOURCE_RUN / "pfsp" / m["file"]).is_file()]
    assert not missing, missing
    print(f"\nexact_eff_ft/latest.pt: step {ck['step']}, {len(a)} agent tensors, optimizers {sorted(oa)} "
          f"({sum(len(oa[n]['state']) for n in oa)} parameter states), scaler {ck['scaler_state_dict']['scale']}, "
          f"pool {len(members)} members; same keys and shapes as {DYNBP_CKPT.parent.name}/{DYNBP_CKPT.name}")
    del ck, ref
    assert (src.stat().st_size, src.stat().st_mtime_ns) == before
