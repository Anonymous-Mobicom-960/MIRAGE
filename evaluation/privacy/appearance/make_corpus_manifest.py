#!/usr/bin/env python3
"""Build the CANONICAL 10 fps corpus shared by every arm, plus its manifest.

The protocol is fixed at 10 fps. Every anonymiser is applied to THIS corpus, so frame indices
correspond 1:1 across the raw control, MIRAGE and the baseline defences, and no arm can be
advantaged by a different temporal sampling. Selection: manifest rows with status==KEPT and
n_identities==1, excluding the background plates. Native resolution is preserved; only the frame
rate changes. CRF 12 is visually lossless and is applied identically to every clip, so any codec
artefact is common-mode across all arms.

Requires the evaluation dataset (distributed separately from this repository), whose root must
contain FLAT_MANIFEST.csv and the source clips.

  python make_corpus_manifest.py --dataset-root <reid_dataset_flat>
"""
import argparse
import csv
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "corpus_10fps"))
    ap.add_argument("--manifest-out",
                    default=os.path.join(HERE, "manifests", "corpus_10fps.json"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.manifest_out), exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(a.dataset_root, "FLAT_MANIFEST.csv"),
                                    encoding="utf-8-sig")))
    kept = [r for r in rows if r["status"] == "KEPT" and r["n_identities"] == "1"
            and not r["identities"].startswith("p00")]
    print(f"[corpus] {len(kept)} clips, {len({r['identities'] for r in kept})} identities",
          flush=True)

    man = []
    for i, r in enumerate(kept, 1):
        src = os.path.join(a.dataset_root, r["filename"])
        dst = os.path.join(a.out_dir, os.path.splitext(r["filename"])[0] + ".mp4")
        if not os.path.exists(src):
            print(f"[corpus] MISSING {src}", flush=True); continue
        if not os.path.exists(dst):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                   "-vf", "fps=10", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                   "-pix_fmt", "yuv420p", "-an", dst]
            subprocess.run(cmd, check=True)
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,nb_frames,r_frame_rate", "-of", "json", dst],
                           capture_output=True, text=True)
        s = json.loads(p.stdout)["streams"][0]
        man.append(dict(clip=os.path.basename(dst), identity=r["identities"],
                        condition=r["condition"], source_relpath=r["source_relpath"],
                        source_file=r["filename"],
                        w=int(s["width"]), h=int(s["height"]),
                        frames=int(s.get("nb_frames", 0)), fps=s["r_frame_rate"]))
        if i % 10 == 0 or i == len(kept):
            print(f"[corpus] {i}/{len(kept)} done", flush=True)

    json.dump(man, open(a.manifest_out, "w"), indent=1)
    tot = sum(m["frames"] for m in man)
    print(f"[corpus] DONE {len(man)} clips, {tot:,} frames @10fps, "
          f"{len({m['identity'] for m in man})} identities", flush=True)


if __name__ == "__main__":
    main()
