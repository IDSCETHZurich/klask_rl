"""GPU-free tests for concurrent scheduling and per-worker device configuration."""

import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scheduler
from run_tournament import gpu_blockers, selected_devices
from worker import configuration_for_job


def test_two_gpus_default_and_explicit_single_gpu():
    assert selected_devices(SimpleNamespace()) == ["cuda:0", "cuda:1"]
    assert selected_devices(SimpleNamespace(device="cuda:1")) == ["cuda:1"]
    assert selected_devices(SimpleNamespace(devices=["cuda:2", "cuda:0"])) == ["cuda:2", "cuda:0"]


@pytest.mark.parametrize("devices", [[], ["cpu"], ["cuda"], ["cuda:-1"], ["cuda:0", "cuda:0"]])
def test_invalid_device_selections_fail(devices):
    with pytest.raises(ValueError):
        selected_devices(SimpleNamespace(devices=devices))


def test_preflight_requires_both_gpus_unless_overridden(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert "cuda:1" in gpu_blockers(["cuda:0", "cuda:1"])[0]
    assert gpu_blockers(["cuda:0"]) == []


def test_job_retargets_nested_model_devices_without_mutating_plan():
    plan = {
        "device": "cuda:0",
        "evaluation_config": {
            "device": "cuda:0",
            "env": {"device": "cuda:0", "seed": 0},
            "model": {
                "device": "cuda:0",
                "sim_device": "cuda:0",
                "train_device": "cuda:0",
                "train_devices": ["cuda:0", "cuda:1"],
                "rssm": {"device": "cuda:0"},
                "encoder": {"mlp": {"device": "cuda:0"}},
                "storage_device": "cpu",
            },
        },
    }
    cfg = configuration_for_job(plan, {"device": "cuda:1", "seed": 7})
    assert cfg.device == cfg.env.device == cfg.model.rssm.device == cfg.model.encoder.mlp.device == "cuda:1"
    assert cfg.model.train_device == cfg.model.sim_device == "cuda:1"
    assert list(cfg.model.train_devices) == ["cuda:1"]
    assert cfg.model.storage_device == "cpu"
    assert cfg.seed == cfg.env.seed == 7
    assert plan["evaluation_config"]["model"]["rssm"]["device"] == "cuda:0"


@pytest.fixture
def process_factory(monkeypatch):
    clock = [0]
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + 1))
    processes = []

    class Process:
        def __init__(self, job):
            self.job = job
            self.ready = clock[0] + job.get("duration", 2)
            self.returncode = None
            self.signals = []
            self.reaped = False

        def poll(self):
            if self.returncode is None and clock[0] >= self.ready:
                self.returncode = self.job.get("returncode", 0)
            return self.returncode

        def wait(self):
            assert self.poll() is not None
            self.reaped = True
            return self.returncode

        def send_signal(self, sig):
            self.signals.append(sig)
            self.returncode = -sig

        def kill(self):
            self.send_signal(signal.SIGKILL)

    def start(job):
        assert not any(p.job["device"] == job["device"] and p.poll() is None for p in processes)
        process = Process(job)
        processes.append(process)
        return process

    return start, processes


def test_runs_concurrently_and_refills_only_the_free_gpu(process_factory):
    start, processes = process_factory
    jobs = [
        {"id": "a0", "device": "cuda:0", "duration": 6},
        {"id": "b0", "device": "cuda:1", "duration": 2},
        {"id": "a1", "device": "cuda:0", "duration": 2},
        {"id": "b1", "device": "cuda:1", "duration": 2},
    ]
    completed = []
    scheduler.run_jobs(jobs, start, lambda job, code: completed.append((job["id"], code)))
    assert [p.job["id"] for p in processes] == ["a0", "b0", "b1", "a1"]
    assert [name for name, code in completed] == ["b0", "b1", "a0", "a1"]
    assert all(code == 0 for _, code in completed)
    assert all(p.reaped and not p.signals for p in processes)
    assert [j["id"] for j in jobs] == ["a0", "b0", "a1", "b1"]


def test_failure_stops_scheduling_and_saves_other_worker(process_factory):
    start, processes = process_factory
    jobs = [
        {"id": "long", "device": "cuda:0", "duration": 100},
        {"id": "failed", "device": "cuda:1", "returncode": 2},
        {"id": "never_start", "device": "cuda:1"},
    ]
    completed = []
    with pytest.raises(scheduler.WorkerFailed, match="failed on cuda:1"):
        scheduler.run_jobs(jobs, start, lambda job, code: completed.append((job["id"], code)))
    assert completed == [("failed", 2)]
    assert len(processes) == 2
    assert processes[0].signals == [signal.SIGINT]
    assert all(p.reaped for p in processes)


def test_interrupt_stops_and_reaps_both_workers(process_factory, monkeypatch):
    start, processes = process_factory

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scheduler.run_jobs(
            [{"id": "a", "device": "cuda:0"}, {"id": "b", "device": "cuda:1"}], start, lambda job, code: None
        )
    assert len(processes) == 2
    assert all(p.signals == [signal.SIGINT] and p.reaped for p in processes)


def test_resumed_pending_jobs_keep_original_gpu_assignments(process_factory):
    start, processes = process_factory
    # The first leg on GPU 0 already completed before interruption.
    pending = [{"id": "leg1", "device": "cuda:1"}, {"id": "leg2", "device": "cuda:0"}]
    scheduler.run_jobs(pending, start, lambda job, code: None)
    assert [(p.job["id"], p.job["device"]) for p in processes] == [("leg1", "cuda:1"), ("leg2", "cuda:0")]


def test_two_real_cpu_subprocesses_can_reach_a_shared_barrier(tmp_path):
    # Each child requires the other to have started. A sequential launcher
    # cannot pass this barrier. No GPU or simulator is needed for this test.
    program = """
import sys, time
from pathlib import Path
root, own, other = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
(root / own).touch()
deadline = time.monotonic() + 10
while not (root / other).exists():
    if time.monotonic() > deadline:
        raise SystemExit(2)
    time.sleep(0.01)
"""
    processes = []

    def start(job):
        with (tmp_path / (job["id"] + ".log")).open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", program, str(tmp_path), job["id"], job["other"]],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes.append(process)
        return process

    completed = []
    scheduler.run_jobs(
        [{"id": "a", "other": "b", "device": "cuda:0"}, {"id": "b", "other": "a", "device": "cuda:1"}],
        start,
        lambda job, code: completed.append((job["id"], code)),
        poll_interval=0.01,
    )
    assert sorted(completed) == [("a", 0), ("b", 0)]
    assert all(p.poll() == 0 for p in processes)
