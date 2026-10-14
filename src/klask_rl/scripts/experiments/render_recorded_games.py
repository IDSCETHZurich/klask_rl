"""Render games recorded with the tournament's `record_states` option to MP4 videos for watching.

Each `<job_dir>/states/env*_game*.npz` holds the seat-0 policy observation per step (sim frame:
seat-0 peg 0:4, seat-1 peg 4:8, ball 8:12). Frames are re-rendered with the sprite renderer at a
higher resolution than the agents see, labelled with the two players, the game clock and, at the
end, the outcome from the match's games.csv. Writes one MP4 per game plus all games in one file.

    python.sh scripts/experiments/render_recorded_games.py --match_dir logs/experiments/watch/drssm57_vs_ppob \
        --output logs/experiments/watch/videos
"""

import argparse
import csv
import glob
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
RENDERER = SCRIPTS / "dreamer" / "sprite_renderer"
sys.path.insert(0, str(RENDERER))
from board_renderer import BoardRenderer  # noqa: E402

OUTCOME = {"term_goal_scored": "{s0} scores", "term_goal_conceded": "{s1} scores", "term_player_in_goal": "{s0} own-goal foul",
           "term_opponent_in_goal": "{s1} own-goal foul", "term_time_out": "time out (draw)"}


def ffmpeg():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def write_mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()


def label(img, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def outcomes(match_dir):
    rows = {}
    for f in glob.glob(f"{match_dir}/*_leg*/games.csv"):
        leg = Path(f).parent.name
        for r in csv.DictReader(open(f)):
            key = next((k for k in OUTCOME if r.get(k) == "1"), "term_time_out")
            rows[(leg, int(r["env_id"]), int(r["episode_id"]))] = key
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=50, help="50 = real time (one frame per 20 ms control step)")
    parser.add_argument("--width", type=int, default=384)
    args = parser.parse_args()
    height = int(round(args.width * 0.42 / 0.32))
    renderer = BoardRenderer(sprite_dir=str(RENDERER / "assets/sprites"), background_path=str(RENDERER / "assets/background/median_background.png"),
                             output_size=(args.width, height), target_frame="sim")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = outcomes(args.match_dir)
    everything = []
    files = sorted(glob.glob(f"{args.match_dir}/*_leg*/states/*.npz"))
    for n, path in enumerate(files, 1):
        d = np.load(path)
        s0, s1 = str(d["seat0"]), str(d["seat1"])
        leg = Path(path).parent.parent.name
        env_id, game = (int(x) for x in Path(path).stem.replace("env", "").split("_game"))
        result = OUTCOME[results.get((leg, env_id, game), "term_time_out")].format(s0=s0, s1=s1)
        frames = []
        for t, st in enumerate(d["state"]):
            img = renderer.render(st[0:2].tolist(), st[4:6].tolist(), st[8:10].tolist()).copy()
            label(img, f"{s1}", (8, 20), color=(120, 200, 255))
            label(img, f"{s0}", (8, height - 10), color=(150, 255, 150))
            label(img, f"game {n}  t = {t * 0.02:5.2f} s", (args.width - 175, 20))
            frames.append(img)
        end = frames[-1].copy()
        label(end, result, (8, height // 2), scale=0.65, color=(0, 255, 255))
        frames += [end] * args.fps  # hold the result for 1 s
        write_mp4(frames, out / f"game{n}_{s0}_vs_{s1}.mp4", args.fps)
        everything += frames
        print(f"game {n}: {s0} (bottom) vs {s1} (top), {len(d['state']) * 0.02:.1f} s, {result}", flush=True)
    if everything:
        write_mp4(everything, out / "all_games.mp4", args.fps)
        print(f"wrote {len(files)} games to {out}")


if __name__ == "__main__":
    main()
