#!/usr/bin/env bash
# Native-offload disk tier backing store (R108 promotion): a fixed-size loopback ext4 image.
# WHY: LMCache's fs_native cap needed patch 0008 after the R69 876G disk-fill; the v0.28 native
# fs tier's cap enforcement is UNVERIFIED — so the cap is enforced BY CONSTRUCTION instead:
# the tier lives on a 200G preallocated image and physically cannot grow past it. A full tier
# degrades to store-failures (engine keeps serving); the root fs is never at risk.
# Idempotent. Run once:  ssh flan 'sudo bash -s' < flan/setup-native-l2.sh
set -euo pipefail
IMG=/srv/qwen5090/native-l2.img
MNT=/srv/qwen5090/native-l2
SIZE=200G
if [ ! -f "$IMG" ]; then
  fallocate -l $SIZE "$IMG"
  mkfs.ext4 -q -m 0 -L native-l2 "$IMG"
  echo "created $IMG ($SIZE)"
fi
mkdir -p "$MNT"
grep -q "native-l2.img" /etc/fstab || echo "$IMG $MNT ext4 loop,noatime 0 0" >> /etc/fstab
mountpoint -q "$MNT" || mount "$MNT"
df -h "$MNT" | tail -1
