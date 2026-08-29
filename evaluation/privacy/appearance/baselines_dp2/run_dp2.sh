#!/usr/bin/env bash
# DeepPrivacy2 baseline inference over the shared 10 fps corpus, run exactly as published:
# the authors' own implementation, unmodified, in their own container, at their FB_cse full-body
# configuration with multi-modal truncation and per-track latent caching.
#
# One-time setup (the pinned commit is the one used for the reported comparison):
#   git clone https://github.com/hukkelas/deep_privacy2.git repo
#   git -C repo checkout f4d8f09d1eb8f758c89cf1795ae41f20533942d3
#   docker build -t dp2:pinned repo
#
# Then, from this directory:
#   CORPUS=../corpus_10fps ./run_dp2.sh
#
# Output lands in ./out/, one anonymised MP4 per corpus clip, plus _RUN_DP2.json recording
# per-clip success/failure. Feed ./out/ to extract_arm.py as the dp2 arm's --video-dir.
set -u
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "$0")" && pwd)"
CORPUS="${CORPUS:-$HERE/../corpus_10fps}"
mkdir -p "$HERE/out" "$HERE/torch_home/.torch"
docker image inspect dp2:pinned >/dev/null 2>&1 || { echo "dp2:pinned image missing; see header"; exit 1; }
echo "[$(date +%H:%M:%S)] starting DeepPrivacy2 inference"
docker run --rm --gpus all --user root \
  -e TORCH_HOME=/root/.cache -e PYTHONUNBUFFERED=1 \
  -v "$CORPUS:/data/in:ro" \
  -v "$HERE/out:/data/out" \
  -v "$HERE/torch_home:/root/.cache" \
  -v "$HERE/torch_home/.torch:/root/.torch" \
  -v "$HERE/repo:/home/testuser/dp2" \
  -v "$HERE/run_corpus.py:/home/testuser/run_corpus.py:ro" \
  -w /home/testuser/dp2 dp2:pinned python /home/testuser/run_corpus.py
echo "DP2_RUN EXIT=$?"
