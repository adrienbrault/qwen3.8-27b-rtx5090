#!/usr/bin/env python3
"""Offline checks of scripts/serve.sh: python3 -m unittest discover -s scripts/tests -p 'test_serve.py'.

Fake docker, nvidia-smi and curl commands stand in for the host; no container starts and no GPU is touched.
"""

import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "scripts/serve.sh"
PCIE_ENV = ("VLLM_SM12X_PCIE_IPC_AR=1", "VLLM_SM12X_PCIE_IPC_MTP=1")

FAKE_DOCKER = r'''#!/usr/bin/env bash
# Records `run` arguments, answers the inspections serve.sh makes, and prints $FAKE_LOG as the container log.
case "$1" in
  image) exit 0;;
  container) exit 1;;                                  # no container of that name exists yet
  run) shift; printf '%s\n' "$@" > "$FAKE_STATE/run-args"; echo cid;;
  inspect) case "$*" in *State.Running*) echo "true 0";; *) cat "$FAKE_STATE/run-args";; esac;;
  logs) cat "$FAKE_LOG";;
  exec) case "$3" in python3) echo "0.31.0 0.7.0.post1";; *) exit 0;; esac;;
  rm|stop) exit 0;;
  *) echo "fake docker: unexpected $*" >&2; exit 2;;
esac
'''

FAKE_NVIDIA_SMI = r'''#!/usr/bin/env bash
case "$*" in
  -L) echo "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-0)"; echo "GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-1)";;
  "topo -p2p r") printf ' \t\033[4mGPU0\tGPU1\t\033[0m\n GPU0\tX\t%s\t\n GPU1\t%s\tX\t\n\nLegend:\n\n  X    = Self\n' "$FAKE_P2P" "$FAKE_P2P";;
  *) echo "fake nvidia-smi: unexpected $*" >&2; exit 2;;
esac
'''

BOOT_LINES = [
    "linear-V-scale store overlay ACTIVE",
    "as specified by kv_cache_memory_bytes",
    "int workspace shrunk 8 MiB -> 1 MiB",
    "Setting attention block size to 1472 tokens",
    "Offloader set to UVAOffloader",
    "Total CPU offloaded parameters: 1.18 GiB",
    "Batch-sharded sampling enabled",
    "re-enabled FlashInfer split-KV",
    "SM12X draft UB plan ACTIVE: VLLM_SM12X_DRAFT_UB_PLAN=1",
    "SM12X draft UB plan USED",
    "GPU KV cache size: 1,490,437 tokens",
]
PCIE_ON_LINES = [
    "PCIe IPC all-reduce enabled: TP2 bf16, max_rows=320",
    "Using ['PCIE_IPC', 'CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp'",
    "SM12X PCIe IPC: MTP drafter admitted",
]
# What a host without peer-to-peer logs: CustomAllreduce fails its P2P test, pcie_ipc refuses without it, NCCL remains.
NO_P2P_LINES = [
    "Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed.",
    "PCIe IPC all-reduce disabled: knob off or stateless group",
    "Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp'",
]


def unused_port():
    """A port in the dynamic range, not bound here: serve.sh only checks that nothing listens on it."""
    for port in range(49152 + os.getpid() % 8000, 65535):
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no unused port")


class ServeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.model = base / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text('{"truncation": null}')
        self.bin = base / "bin"
        self.bin.mkdir()
        for name, body in (("docker", FAKE_DOCKER), ("nvidia-smi", FAKE_NVIDIA_SMI),
                           ("curl", "#!/usr/bin/env bash\nexit 0\n")):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        self.state = base / "state"
        self.state.mkdir()
        self.log = base / "container.log"
        self.env = {k: v for k, v in os.environ.items() if k not in ("PCIE_IPC", "BA4_INPUTS", "KV_TIER_DIR")}
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}", MODEL_DIR=str(self.model),
                        CACHE_DIR=str(base / "cache"), SERVE_ENV=str(base / "serve.env"),
                        PORT=str(unused_port()), FAKE_STATE=str(self.state), FAKE_LOG=str(self.log),
                        FAKE_P2P="OK", IMAGE="vllm-qwen38:v0310-r27c-A_S-BA4-L1a-ubplan")
        (base / "serve.env").write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def serve(self, *args, **env):
        return subprocess.run(["bash", str(SERVE), *args], env={**self.env, **env}, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

    def run_args(self):
        return (self.state / "run-args").read_text()

    def test_print_default_passes_both_pcie_ipc_variables(self):
        r = self.serve("--print")
        self.assertEqual(r.returncode, 0, r.stderr)
        for v in PCIE_ENV:
            self.assertIn(v, r.stdout)
        self.assertIn("NCCL_P2P_LEVEL=SYS", r.stdout)
        self.assertIn("--kv-cache-memory-bytes 15900000000", r.stdout)
        self.assertNotIn("pcie_ipc all-reduce off", r.stderr)

    def test_print_pcie_ipc_off_passes_neither_variable(self):
        r = self.serve("--print", PCIE_IPC="0")
        self.assertEqual(r.returncode, 0, r.stderr)
        for v in PCIE_ENV:
            self.assertNotIn(v, r.stdout)
        self.assertIn("NCCL_P2P_LEVEL=SYS", r.stdout)
        self.assertIn("pcie_ipc all-reduce off (PCIE_IPC=0)", r.stderr)

    def test_settings_file_value_applies(self):
        Path(self.env["SERVE_ENV"]).write_text("PCIE_IPC=0\n")
        r = self.serve("--print")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn(PCIE_ENV[0], r.stdout)

    def test_invalid_value_is_refused(self):
        r = self.serve("--print", PCIE_IPC="2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("PCIE_IPC must be 0 or 1", r.stderr)

    def test_served_boot_passes_with_pcie_ipc_lines(self):
        self.log.write_text("\n".join(BOOT_LINES + PCIE_ON_LINES) + "\n")
        r = self.serve()
        self.assertEqual(r.returncode, 0, r.stderr)
        for v in PCIE_ENV:
            self.assertIn(v, self.run_args())
        self.assertIn("all-reduce PCIE_IPC,CUSTOM,PYNCCL", r.stdout)

    def test_served_boot_without_pcie_ipc_lines_fails(self):
        self.log.write_text("\n".join(BOOT_LINES + NO_P2P_LINES) + "\n")
        r = self.serve()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("boot log lacks: PCIe IPC all-reduce enabled", r.stderr)
        self.assertIn("boot log lacks: SM12X PCIe IPC: MTP drafter admitted", r.stderr)

    def test_missing_peer_to_peer_stops_before_the_container_starts(self):
        r = self.serve(FAKE_P2P="CNS")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("reports 'CNS' for GPU 0 -> GPU 1", r.stderr)
        self.assertIn("PCIE_IPC=0", r.stderr)
        self.assertFalse((self.state / "run-args").exists())

    def test_pcie_ipc_off_boots_without_peer_to_peer(self):
        self.log.write_text("\n".join(BOOT_LINES + NO_P2P_LINES) + "\n")
        r = self.serve(PCIE_IPC="0", FAKE_P2P="CNS")
        self.assertEqual(r.returncode, 0, r.stderr)
        for v in PCIE_ENV:
            self.assertNotIn(v, self.run_args())
        self.assertIn("all-reduce PYNCCL", r.stdout)

    def test_pcie_ipc_off_fails_when_the_boot_shows_it_enabled(self):
        self.log.write_text("\n".join(BOOT_LINES + PCIE_ON_LINES) + "\n")
        r = self.serve(PCIE_IPC="0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("boot log shows: PCIe IPC all-reduce enabled", r.stderr)


if __name__ == "__main__":
    unittest.main()
