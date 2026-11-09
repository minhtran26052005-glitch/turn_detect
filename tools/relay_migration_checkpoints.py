"""Stream frozen Colab checkpoints to the SSH staging directory, with SHA checks."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import quote

import requests
from colab_runtime import ColabRuntime, assigned_servers

HOST = 'edabk@100.91.209.124'
ROOT = '/home/edabk/hoangbpm/turn_detect/staging/checkpoints'
RECEIPT = Path('outputs/translation_segmenter_v15d_20261026/ssh_migration_v1/frozen_checkpoint_receipt.json')
tls = threading.local()


def ssh(command, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST, command],
                          check=True, stdin=subprocess.DEVNULL, timeout=120, **kwargs)


def runtime():
    if not hasattr(tls, 'r'):
        notebook = json.loads(RECEIPT.read_text(encoding='utf-8'))['connection']['notebook']
        for index in range(len(assigned_servers())):
            candidate = ColabRuntime(index, session_path=notebook)
            try:
                candidate.connect()
                tls.r = candidate
                break
            except Exception:
                candidate.close()
        else:
            raise RuntimeError('Frozen checkpoint session is unavailable; reconnect Colab')
    return tls.r


def transfer(pair):
    lang, info = pair
    sha, size = info['sha256'], info['bytes']
    assert lang in ('vi', 'en', 'zh') and re.fullmatch(r'[0-9a-f]{64}', sha)
    remote = f'{ROOT}/{lang}_{sha}.pt.partial'
    completed = f'{ROOT}/{lang}_{sha}.pt'
    p = ssh(f'test -f {completed} && stat -c %s {completed} || echo 0', capture_output=True, text=True)
    if int(p.stdout.strip()) == size:
        got = ssh(f'sha256sum {completed}', capture_output=True, text=True).stdout.split()[0]
        if got != sha:
            raise RuntimeError(f'{lang}: existing checkpoint SHA mismatch')
        print(lang, 'already verified', flush=True)
        return
    p = ssh(f'test -f {remote} && stat -c %s {remote} || echo 0', capture_output=True, text=True)
    offset = int(p.stdout.strip())
    if offset > size:
        raise RuntimeError(f'{lang}: partial checkpoint is too large')
    print(f'{lang}: resuming from {offset / 1024**2:.0f} MiB', flush=True)
    writer = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', HOST, f'cat >> {remote}'], stdin=subprocess.PIPE)
    written, failures, next_report = offset, 0, offset // (256 * 1024**2)
    try:
        while written < size:
            end = min(size, written + 16 * 1024**2) - 1
            response = None
            try:
                response = runtime().request('GET', 'files/' + quote(info['path'].lstrip('/'), safe='/'),
                    headers={'Range': f'bytes={written}-{end}'}, stream=True)
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {written}-{end}/{size}':
                    raise RuntimeError(f'{lang}: unexpected HTTP range')
                for chunk in response.iter_content(1024**2):
                    if chunk:
                        writer.stdin.write(chunk)
                        written += len(chunk)
                if written != end + 1:
                    raise RuntimeError(f'{lang}: incomplete HTTP range')
                failures = 0
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                failures += 1
                if failures > 5:
                    raise RuntimeError(f'{lang}: repeated connection errors; partial preserved') from exc
                print(f'{lang}: reconnecting at {written / 1024**2:.0f} MiB', flush=True)
                time.sleep(min(2 * failures, 10))
                continue
            finally:
                if response is not None:
                    response.close()
            if written // (256 * 1024**2) > next_report:
                next_report = written // (256 * 1024**2)
                check = ssh(f'df -Pk {ROOT}', capture_output=True, text=True).stdout.splitlines()[-1].split()
                if int(check[3]) * 1024 < 10 * 1024**3:
                    raise RuntimeError('Remote SSD reached 10 GiB reserve')
                print(f'{lang}: {written / 1024**2:.0f}/{size / 1024**2:.0f} MiB', flush=True)
    finally:
        writer.stdin.close()
        rc = writer.wait()
        if rc:
            raise RuntimeError(f'{lang}: SSH stream failed; partial preserved')
    got = ssh(f'sha256sum {remote}', capture_output=True, text=True).stdout.split()[0]
    if got != sha:
        raise RuntimeError(f'{lang}: checkpoint SHA mismatch; partial preserved')
    ssh(f'test ! -e {completed} && mv -- {remote} {completed}')
    print(lang, 'checkpoint SHA-256 verified', flush=True)


def main():
    global RECEIPT
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--languages', nargs='+', choices=('vi', 'en', 'zh'), default=['vi', 'en', 'zh'])
    args = parser.parse_args()
    RECEIPT = args.receipt
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))
    ssh(f'mkdir -p -- {ROOT}')
    result = ssh(f'df -Pk {ROOT}', capture_output=True, text=True).stdout.splitlines()[-1].split()
    selected = [(lang, receipt['files'][lang]) for lang in args.languages]
    if int(result[3]) * 1024 < sum(x['bytes'] for _, x in selected) + 15 * 1024**3:
        raise RuntimeError('Remote SSD does not have space for checkpoints and reserve')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(transfer, selected))
    print('Selected frozen checkpoints verified on SSH host', flush=True)


if __name__ == '__main__':
    main()
