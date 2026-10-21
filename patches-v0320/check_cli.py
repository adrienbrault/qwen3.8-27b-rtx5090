#!/usr/bin/env python3
"""Check the installed v0.32 parser using CPU-only SM120 metadata, without booting."""
from pathlib import Path
import importlib.metadata

from plan import command

assert importlib.metadata.version('vllm') == '0.32.0'
import vllm.platforms as platforms
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum


class ParserMetadata(Platform):
    _enum = PlatformEnum.CUDA
    device_type = 'cuda'
    device_name = 'cuda'
    dispatch_key = 'CUDA'
    device_control_env_var = 'CUDA_VISIBLE_DEVICES'

    @classmethod
    def get_device_capability(cls, device_id=0):
        return DeviceCapability(12, 0)


# Only schema/default resolution is tested. Importing the actual CUDA platform
# would require libcuda on the CPU builder; no driver or CUDA kernels are used.
platforms._current_platform = ParserMetadata()
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

cmd = command(Path('/model'), Path('/state/v032-candidate'))
args = cmd[cmd.index('vllm.entrypoints.openai.api_server') + 1:]
parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(args)
assert parsed.tensor_parallel_size == 2
assert parsed.attention_backend == "FLASHINFER"
assert parsed.max_num_seqs == 16
assert parsed.kv_cache_memory_bytes == 15900000000
assert parsed.speculative_config['disable_eagle_block_drop'] is True
print('Actual v0.32 parser accepted the candidate flags; CPU metadata only, no server startup')
