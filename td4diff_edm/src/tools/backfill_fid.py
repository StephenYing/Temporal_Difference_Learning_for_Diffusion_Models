#!/usr/bin/env python3
import os
import re
import json
import glob
import time
import argparse
import subprocess
from typing import List, Dict


def list_candidate_runs(base_dir: str) -> List[str]:
    entries = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and (name.startswith('results') or name.startswith('training-runs')):
            entries.append(p)
    return entries


def parse_fid_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_fid_jsonl(path: str, rows: List[Dict]):
    rows_sorted = sorted(rows, key=lambda r: (int(r.get('tick', -1)), float(r.get('kimg', 0.0))))
    with open(path, 'w') as f:
        for r in rows_sorted:
            f.write(json.dumps(r) + '\n')


def find_fid_snapshot_ticks(run_dir: str) -> List[int]:
    ticks = []
    for p in glob.glob(os.path.join(run_dir, 'network-snapshot-fid-*.pkl')):
        m = re.search(r'network-snapshot-fid-(\d{6})\.pkl$', p)
        if m:
            ticks.append(int(m.group(1)))
    return sorted(set(ticks))


def choose_runs_interactive(base_dir: str) -> List[str]:
    runs = list_candidate_runs(base_dir)
    if not runs:
        print('No candidate result directories found.')
        return []
    print('Select run directories to backfill FID (comma-separated indices):')
    for idx, p in enumerate(runs):
        print(f'  [{idx}] {p}')
    sel = input('Your choice: ').strip()
    indices = []
    for part in sel.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except Exception:
            pass
    chosen = [runs[i] for i in indices if 0 <= i < len(runs)]
    return chosen


def infer_time_sec_for_tick(existing: List[Dict], tick: int) -> float:
    # Try linear interpolation based on neighbor ticks
    prev = None
    nxt = None
    for r in sorted(existing, key=lambda r: int(r.get('tick', -1))):
        t = int(r.get('tick', -1))
        if t < tick:
            prev = r
        elif t > tick and nxt is None:
            nxt = r
            break
    if prev is not None and nxt is not None:
        t0, s0 = int(prev['tick']), float(prev.get('time_sec', 0.0))
        t1, s1 = int(nxt['tick']), float(nxt.get('time_sec', s0))
        if t1 != t0:
            return s0 + (s1 - s0) * (tick - t0) / (t1 - t0)
    if prev is not None:
        return float(prev.get('time_sec', 0.0))
    if nxt is not None:
        return float(nxt.get('time_sec', 0.0))
    return 0.0


def run_generate(gen_outdir: str, network_pkl: str, seeds_num: int, batch: int, master_port: int):
    os.makedirs(gen_outdir, exist_ok=True)
    last_seed = max(seeds_num - 1, 1)
    cmd = [
        'python', 'generate.py',
        f'--outdir={gen_outdir}', f'--seeds=0-{last_seed}', '--subdirs', f'--network={network_pkl}', f'--batch={batch}'
    ]
    env = os.environ.copy()
    env['MASTER_ADDR'] = '127.0.0.1'
    env['MASTER_PORT'] = str(master_port)
    env['WORLD_SIZE'] = '1'
    env['RANK'] = '0'
    env['LOCAL_RANK'] = '0'
    subprocess.run(cmd, env=env, check=True)


def run_fid(images_dir: str, ref_path: str, num: int, batch: int) -> float:
    # Use torchrun fid.py for robustness
    cmd = [
        'torchrun', '--standalone', '--nproc_per_node=1', 'fid.py', 'calc',
        f'--images={images_dir}', f'--ref={ref_path}', f'--num={num}', f'--batch={batch}'
    ]
    out = subprocess.check_output(' '.join(cmd), shell=True, text=True)
    # Parse float from output (last number)
    tokens = [t for t in out.strip().split() if re.match(r'^\d+(?:\.\d+)?$', t)]
    if not tokens:
        # As a fallback, try to read from stdout lines
        lines = [l for l in out.strip().splitlines() if l.strip()]
        for l in reversed(lines):
            m = re.search(r'([0-9]+\.[0-9]+)$', l.strip())
            if m:
                return float(m.group(1))
        raise RuntimeError('Failed to parse FID from output:\n' + out)
    return float(tokens[-1])


def backfill_run(run_dir: str, start_tick: int = None, end_tick: int = None, gen_batch: int = 32):
    print(f'>>> Backfilling: {run_dir}')
    opts_path = os.path.join(run_dir, 'training_options.json')
    fid_jsonl = os.path.join(run_dir, 'fid.jsonl')
    fid_rows = parse_fid_jsonl(fid_jsonl)
    existing_ticks = set(int(r.get('tick', -1)) for r in fid_rows if 'tick' in r)

    # Load options for fid parameters
    fid_num = 10000
    fid_ref = 'fid-refs/cifar10-32x32.npz'
    fid_batch = 64
    try:
        with open(opts_path, 'r') as f:
            opts = json.load(f)
            fid_num = int(opts.get('fid_num_images', fid_num))
            fid_ref = str(opts.get('fid_ref_path', fid_ref))
            fid_batch = int(opts.get('fid_max_batch', fid_batch))
    except Exception:
        pass

    # Find available fid snapshots
    fid_ticks = find_fid_snapshot_ticks(run_dir)
    if start_tick is not None or end_tick is not None:
        fid_ticks = [t for t in fid_ticks if (start_tick is None or t >= start_tick) and (end_tick is None or t <= end_tick)]

    missing = [t for t in fid_ticks if t not in existing_ticks]
    if not missing:
        print('No missing FID ticks found.')
        return

    print(f'Missing ticks: {missing}')
    for idx, tick in enumerate(missing, 1):
        pkl_path = os.path.join(run_dir, f'network-snapshot-fid-{tick:06d}.pkl')
        if not os.path.isfile(pkl_path):
            print(f'  [skip] snapshot not found for tick {tick}: {pkl_path}')
            continue
        gen_dir = os.path.join(run_dir, f'fid-tmp-tick{tick}')
        try:
            print(f'  [{idx}/{len(missing)}] Generating images for tick {tick}...')
            run_generate(gen_outdir=gen_dir, network_pkl=pkl_path, seeds_num=fid_num, batch=gen_batch, master_port=29650 + (tick % 100))
            print(f'  [{idx}/{len(missing)}] Computing FID for tick {tick}...')
            fid_val = run_fid(images_dir=gen_dir, ref_path=fid_ref, num=fid_num, batch=fid_batch)
        except subprocess.CalledProcessError as e:
            print(f'  [error] external command failed at tick {tick}: {e}')
            continue
        except Exception as e:
            print(f'  [error] failed at tick {tick}: {e}')
            continue

        # Fill a row and insert in order
        row = {
            'tick': int(tick),
            'kimg': float(tick * 50.0),  # approximate; training logs use kimg per tick, but we cannot reconstruct exactly
            'fid': float(fid_val),
            'time_sec': float(infer_time_sec_for_tick(fid_rows, tick)) or float(time.time())
        }
        fid_rows.append(row)
        write_fid_jsonl(fid_jsonl, fid_rows)
        print(f'  [+] tick {tick} FID={fid_val:.3f} written to {fid_jsonl}')


def main():
    parser = argparse.ArgumentParser(description='Backfill missing FID evaluations for training runs.')
    parser.add_argument('--runs', nargs='*', help='Run directories to process')
    parser.add_argument('--start-tick', type=int, default=None)
    parser.add_argument('--end-tick', type=int, default=None)
    parser.add_argument('--gen-batch', type=int, default=32, help='Batch for generate.py during backfill')
    args = parser.parse_args()

    if not args.runs:
        chosen = choose_runs_interactive(os.getcwd())
    else:
        chosen = args.runs

    if not chosen:
        print('No runs selected. Exiting.')
        return

    for run_dir in chosen:
        backfill_run(run_dir, start_tick=args.start_tick, end_tick=args.end_tick, gen_batch=args.gen_batch)


if __name__ == '__main__':
    main()


