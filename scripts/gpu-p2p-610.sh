#!/usr/bin/env bash
# flan GPU driver 595.84 -> 610.57.04 + QuixiAI P2P kernel modules (user 2026-08-31:
# "Yes update gpu driver with p2p"). Dual RTX 5090 on X870 Taichi Creator; stock driver
# reports P2P "CNS"; the QuixiAI fork (tinygrad P2P patch successor, Blackwell-native)
# unblocks BAR1 P2P — prereqs already true here: open-flavor driver, BAR1=32G (full FB).
#   Fork pin: QuixiAI/open-gpu-kernel-modules @ 7493f4cd63b0 (2026-08-27)
#   Userspace: NVIDIA-Linux-x86_64-610.57.04.run (download.nvidia.com, ~463MB)
#
# PHASES (run on flan as adrienbrault):
#   prep    — verify downloads + checkout pin (safe while daily serves)
#   apply   — MAINTENANCE WINDOW, run while ON kernel 7.0.0-30: stops GPU containers,
#             purges apt nvidia driver stack (no DKMS shadowing), installs 610 userspace
#             (--no-kernel-modules), builds+installs patched modules for the RUNNING
#             kernel, sets GRUB (iommu=pt; drops X570-era quirks), then REBOOTS.
#   verify  — post-reboot: driver 610.57.04 loaded, topo -p2p = OK, torch peer check
#             inside the vLLM image, then relaunch daily.
# ROLLBACK: sudo apt-get install -y nvidia-driver-595-open nvidia-dkms-595-open
#           && sudo nvidia-uninstall -s; GRUB line is backed up as /etc/default/grub.p2p.bak
set -uo pipefail
D=/srv/qwen5090/drivers
RUN=$D/NVIDIA-Linux-x86_64-610.57.04.run
SRC=$D/open-gpu-kernel-modules
PIN=7493f4cd63b0
PHASE=${1:-prep}
log(){ echo "$(date -Is) [gpu-p2p] $*"; }

case "$PHASE" in
prep)
  [ -f "$RUN" ] && [ "$(stat -c%s "$RUN")" -eq 463025450 ] || { log "run file missing/short: $(stat -c%s "$RUN" 2>/dev/null || echo 0)/463025450"; exit 1; }
  [ -d "$SRC" ] || { log "clone missing"; exit 1; }
  git -C "$SRC" fetch -q origin && git -C "$SRC" checkout -q "$PIN" || { log "checkout $PIN failed"; exit 1; }
  chmod +x "$RUN"
  log "prep OK: run file + fork @ $(git -C "$SRC" rev-parse --short HEAD)"
  ;;
apply)
  [ "$(uname -r)" = "7.0.0-30-generic" ] || { log "REFUSING: not on 7.0.0-30-generic (uname -r=$(uname -r)) — reboot into the HWE kernel first"; exit 1; }
  bash "$0" prep || exit 1
  log "stopping GPU consumers"
  timeout 120 sudo docker stop vllm-27b vllm-exp vllm-eval dcgm-exporter 2>/dev/null
  sudo systemctl stop nvidia-persistenced 2>/dev/null
  sudo fuser -k /dev/nvidia* 2>/dev/null; sleep 3
  log "purging apt nvidia driver stack (container-toolkit untouched)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get remove --purge -y 'nvidia-driver-*' 'nvidia-dkms-*' 'nvidia-kernel-*' 'nvidia-utils-*' 'libnvidia-*' 'nvidia-compute-utils-*' 'nvidia-firmware-*' 'xserver-xorg-video-nvidia-*' 2>&1 | tail -2
  # the libnvidia-* glob above also sweeps libnvidia-container* -> docker loses GPU support;
  # reinstall the container toolkit stack (its apt source survives the purge)
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1 2>&1 | tail -1
  sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null
  log "installing 610.57.04 userspace (no kernel modules)"
  sudo "$RUN" -s --no-kernel-modules --no-x-check 2>&1 | tail -3
  log "building patched modules for $(uname -r)"
  ( cd "$SRC" && make modules -j"$(nproc)" >"$D/build.log" 2>&1 && sudo make modules_install >>"$D/build.log" 2>&1 && sudo depmod ) || { log "BUILD FAILED — see $D/build.log tail:"; tail -5 "$D/build.log"; exit 1; }
  # no DKMS left after purge, so no updates/dkms shadowing; assert anyway
  M=$(modinfo -n nvidia 2>/dev/null); case "$M" in *updates/dkms*) log "SHADOWED by $M — aborting"; exit 1;; esac
  bash "$0" fix-nouveau noreboot
  log "GRUB: iommu=pt, dropping X570-era quirks (backup at grub.p2p.bak)"
  sudo cp /etc/default/grub /etc/default/grub.p2p.bak
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="nvidia-drm.modeset=1 iommu=pt"/' /etc/default/grub
  sudo update-grub 2>&1 | tail -1
  log "rebooting in 10s (ctrl-c to abort)"; sleep 10; sudo reboot
  ;;
fix-nouveau)
  # apt purge removes the nvidia packages' nouveau blacklist; without it nouveau binds the
  # GPUs at boot and nvidia.ko probes nothing ("already bound to nouveau"). Blacklist +
  # initramfs rebuild; reboots unless $2=noreboot (apply calls it inline pre-reboot).
  printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
  # the purge also drops Ubuntu's nvidia options file; without NVreg_EnableResizableBar=1 the
  # driver's BAR1-static P2P path is off and P2P falls into the gm200 mailbox path ->
  # NVRM kern_bus assertions + silently corrupt peer copies (seen 2026-08-31).
  printf 'options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords=RMForceStaticBar1=1\n' | sudo tee /etc/modprobe.d/nvidia-options.conf >/dev/null
  sudo update-initramfs -u 2>&1 | tail -1
  log "nouveau blacklisted + NVreg_EnableResizableBar=1, initramfs rebuilt"
  [ "${2:-}" = "noreboot" ] || { log "rebooting in 5s"; sleep 5; sudo reboot; }
  ;;
fix-params)
  # apply the modprobe options to the RUNNING system: stop GPU users, reload modules
  printf 'options nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords=RMForceStaticBar1=1\n' | sudo tee /etc/modprobe.d/nvidia-options.conf >/dev/null
  timeout 120 sudo docker stop vllm-27b vllm-exp vllm-eval dcgm-exporter 2>/dev/null
  sudo systemctl stop nvidia-persistenced 2>/dev/null
  sudo fuser -k /dev/nvidia* 2>/dev/null; sleep 3
  sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null
  sudo modprobe nvidia NVreg_EnableResizableBar=1 NVreg_RegistryDwords=RMForceStaticBar1=1 && sudo modprobe nvidia_uvm && sudo modprobe nvidia_modeset && sudo modprobe nvidia_drm
  grep -iE "ResizableBar" /proc/driver/nvidia/params
  # NOTE 2026-08-31: RMForceStaticBar1=1 is the key that actually engages the BAR1-static
  # P2P path on the 5090 pair here; EnableResizableBar=1 alone still corrupts (mailbox path).
  ;;
verify)
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" = "610.57.04" ] || { log "driver mismatch"; exit 1; }
  nvidia-smi topo -p2p r
  nvidia-smi topo -p2p r 2>/dev/null | grep -aq OK || { log "P2P still not OK"; exit 1; }
  sudo docker run --rm --gpus all --entrypoint python3 vllm-qwen38:v0280-nvfp4kv - <<'PY'
import torch
a = torch.cuda.can_device_access_peer(0, 1); b = torch.cuda.can_device_access_peer(1, 0)
print("torch peer access 0<->1:", a, b)
assert a and b
x = torch.ones(1024, 1024, device="cuda:0"); y = x.to("cuda:1"); assert y.sum().item() == 1024*1024
print("peer copy OK")
PY
  log "verify OK — relaunching daily"
  env -i HOME="$HOME" USER="$USER" PATH="$PATH" bash /srv/qwen5090/launch-daily.sh 2>&1 | grep -aE "DAILY UP|FAILED|KV pool" | cut -c1-160
  ;;
*) log "usage: $0 prep|apply|verify"; exit 1;;
esac
