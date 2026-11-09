"""Isolated optional VAD dependencies, then public causal-ASR preparation."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root=Path('/content/turn_detect_v2')
    target=root/'.deps-v5-vad'
    # VAD uses ORT CPU. These modules are project-local and do not replace the
    # Colab environment, Torch, Transformers, NumPy or the running kernel.
    subprocess.run([sys.executable,'-m','pip','install','--target',str(target),'--no-deps',
        'faster-whisper==1.2.1','ctranslate2==4.6.3','av==16.1.0'],check=True)
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(target)+os.pathsep+str(root)+(os.pathsep+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    subprocess.run([sys.executable,'-u','-m','turn_v2.natural_v5_data','--output','data_v5/natural_20261025',
        '--train-turns','300','--dev-turns','75','--sealed-turns','75','--vi-seconds','900','--asr-batch','8'],
        cwd=root,env=environment,check=True)


if __name__=='__main__':main()
