"""Run independent legs concurrently, with at most one worker per assigned GPU."""

import signal
import time


class WorkerFailed(RuntimeError):
    pass


def stop_workers(processes, timeout=30):
    """Give all workers one shared grace period to save their partial games."""
    processes = list(processes)
    try:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + timeout
        while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        # Also reap every process on a second Ctrl-C during graceful shutdown.
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait()


def run_jobs(jobs, start, finished, *, poll_interval=0.2):
    """Use manifest assignments, so completion order/resume never changes GPUs.

    start(job) returns a Popen-compatible process; finished(job, returncode) is
    invoked serially in the parent, which remains the only report writer.
    """
    pending = list(jobs)
    active = {}
    try:
        while pending or active:
            # Process exits before scheduling replacements: a failure stops all
            # further launches, while the other GPU gets time to save its games.
            failures = []
            for device, (job, process) in list(active.items()):
                returncode = process.poll()
                if returncode is not None:
                    process.wait()
                    del active[device]
                    finished(job, returncode)
                    if returncode:
                        failures.append(f"{job['id']} on {device} exited with code {returncode}")
            if failures:
                raise WorkerFailed("; ".join(failures))
            for job in list(pending):
                if job["device"] not in active:
                    active[job["device"]] = (job, start(job))
                    pending.remove(job)
            if active:
                time.sleep(poll_interval)
    finally:
        stop_workers(process for _, process in active.values())
