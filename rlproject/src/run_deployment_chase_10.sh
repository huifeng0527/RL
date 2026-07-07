#!/usr/bin/env bash
# set -euo pipefail

SUBJECT_PREFIX="${1:-pilot01}"
SECONDS_PER_ROLLOUT="${2:-30}"

for i in $(seq -w 1 10); do
  python rlproject/src/record_deployment_chase.py \
    --subject "${SUBJECT_PREFIX}_r${i}" \
    --seconds "${SECONDS_PER_ROLLOUT}" \
    --save-video

done
