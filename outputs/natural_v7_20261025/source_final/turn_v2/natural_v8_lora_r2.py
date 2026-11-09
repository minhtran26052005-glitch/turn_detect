"""Recover BF16 LoRA from an incompatible, unused Colab torchao probe.

No installed package or base file is changed. The original failed run remains
untouched; all new artifacts live under a new run name.
"""
import argparse
import importlib.metadata
from pathlib import Path

from .common import file_hash,write_json


def disable_unused_torchao():
    import peft.import_utils as peft_imports
    import peft.tuners.lora.torchao as lora_torchao
    peft_imports.is_torchao_available=lambda:False
    lora_torchao.is_torchao_available=lambda:False


def main(args):
    disable_unused_torchao()
    from . import natural_v8_lora as original
    write_json(Path('outputs/natural_v8_20261025/r2_compatibility.json'),{
        'reason':'PEFT0.21.2 rejects installed torchao0.10.0; BF16 nonquantized LoRA does not use torchao',
        'patch_scope':'This child process only; two optional availability probes return false',
        'packages_changed':False,'original_failed_run_preserved':True,'precision':'BF16 base; standard LoRA adapters',
        'peft':importlib.metadata.version('peft'),'torchao':importlib.metadata.version('torchao'),
        'wrapper_sha256':file_hash(Path(__file__)),'training_source_sha256':file_hash(Path(original.__file__))})
    original.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/natural_v8_lora_20261025_r2'))
    main(parser.parse_args())
