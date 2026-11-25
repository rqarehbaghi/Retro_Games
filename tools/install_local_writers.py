"""Install the bounded local writer shortlist in WSL, retaining C: disk space."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume downloads in an existing output folder')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    disk = Path('/mnt/c')
    if not disk.is_dir():
        parser.error('Run in WSL with C: mounted at /mnt/c')
    # Host free space is authoritative: WSL virtual-disk allocation and other
    # Windows processes can consume space beyond the model payload sizes.
    floor = 60 * 1024**3
    for model in ('qwen3.8:27b', 'gemma4:31b', 'gemma4:12b'):
        if shutil.disk_usage(disk).free < floor + 22 * 1024**3:
            raise RuntimeError('Insufficient space for next model plus reserve')
        print('Installing', model, flush=True)
        with (args.out_dir / (model.replace(':', '_') + '.log')).open('a') as log:
            child = subprocess.Popen(['ollama', 'pull', model], stdout=log, stderr=log)
            try:
                while child.poll() is None:
                    if shutil.disk_usage(disk).free < floor:
                        raise RuntimeError('Disk budget reached; stopping download')
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError('Download failed: ' + model + '; see log')
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        print('Installed', model, flush=True)
    subprocess.run(['ollama', 'list'], check=True)
    print('C: free GiB:', round(shutil.disk_usage(disk).free / 1024**3, 1), flush=True)
    here = Path(__file__).resolve().parent
    comparison = args.out_dir / 'comparison'
    if comparison.exists():
        comparison = args.out_dir / ('comparison_' + str(time.time_ns()))
    subprocess.run([sys.executable, str(here / 'compare_local_writers.py'),
                    '--facts', str(here / 'local_writer_sample.txt'),
                    '--out-dir', str(comparison)], check=True)


if __name__ == '__main__':
    main()
