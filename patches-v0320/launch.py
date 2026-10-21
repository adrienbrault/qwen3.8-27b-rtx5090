#!/usr/bin/env python3
"""Launch the candidate only, from inside the GPU-locked screen unit."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
from plan import command

if os.readlink('/proc/self/fd/9') != '/srv/qwen5090/gpu-exclusive.lock':
    raise SystemExit('GPU lock must be inherited on fd 9')
cmd = command(Path('/srv/qwen5090/models/qwen3.8-27b-nvidia-nvfp4'),
              Path('/srv/qwen5090/v032-candidate'))
mnbt = int(os.environ.get('V032_MNBT', '8192'))
if mnbt not in (2048, 4096, 8192):
    raise SystemExit('unsupported screen prefill budget')
cmd[cmd.index('--max-num-batched-tokens')+1] = str(mnbt)
cmd.insert(2, '-d')
Path(sys.argv[1]).write_text(shlex.join(cmd) + '\n')
subprocess.run(cmd, check=True)
