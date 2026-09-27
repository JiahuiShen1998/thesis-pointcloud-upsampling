#!/usr/bin/env python3
"""PointRCNN val inference + official KITTI C++ AP evaluation for one input variant.

Reuses the validated local pattern from
results/rpn4096_no_distance_propose_comparison/run_comparison_method.py,
but keeps the STANDARD default.yaml config (only RPN.LOC_XZ_FINE False override),
matching the checkpoint/config precedent recorded in
results/pointrcnn_clean_original_baseline_20260508_203028/config/command_used.txt
and documented in KITTI_BASELINE_RESULTS.md.

Does not train the detector, does not modify final_bin content, does not touch
upsampling code. Only swaps the data/KITTI/object/training/velodyne symlink for
the duration of one inference run, then restores it.
"""
from pathlib import Path
import argparse, csv, json, os, shutil, subprocess, sys, time

REPO = Path('/home/ra87racy/projects/baseline_detectors/PointRCNN')
PYTHON = REPO / 'venv_pointrcnn/bin/python'
TOOLS = REPO / 'tools'
VAL = REPO / 'data/KITTI/ImageSets/val.txt'
TRAIN = REPO / 'data/KITTI/object/training'
VEL = TRAIN / 'velodyne'
CPP_EVAL = REPO / 'kitti_devkit/cpp/build/evaluate_object'
LABEL_FULL = REPO / 'kitti_devkit/cpp/data/object/label_2_eval_full'
LABEL_REAL = REPO / 'data/KITTI/object/training/label_2'
CKPT = TOOLS / 'PointRCNN.pth'
CFG = TOOLS / 'cfgs/default.yaml'


def frame_ids(split_file):
    return [x.strip() for x in Path(split_file).read_text().splitlines() if x.strip()]


def record_velodyne(path):
    if path.is_symlink():
        return os.readlink(path)
    if path.exists():
        return 'EXISTS_NON_SYMLINK'
    return 'ABSENT'


def restore_velodyne(before):
    if before == 'ABSENT':
        if VEL.is_symlink() or VEL.exists():
            if VEL.is_dir() and not VEL.is_symlink():
                raise RuntimeError('Refusing to remove non-symlink velodyne directory')
            VEL.unlink(missing_ok=True)
    elif before == 'EXISTS_NON_SYMLINK':
        raise RuntimeError('Original velodyne was non-symlink; refusing automatic restore')
    else:
        if VEL.is_symlink() or VEL.exists():
            if VEL.is_dir() and not VEL.is_symlink():
                raise RuntimeError('Refusing to replace non-symlink velodyne directory')
            VEL.unlink(missing_ok=True)
        VEL.symlink_to(before)


def run_inference(name, input_dir, out_dir, split_file):
    out_dir.mkdir(parents=True, exist_ok=True)
    before = record_velodyne(VEL)
    (out_dir / 'velodyne_before.txt').write_text(before + '\n')

    # eval_rcnn_official.py always reads split file named ImageSets/<TEST.SPLIT>.txt
    split_name = Path(split_file).stem

    command = [
        str(PYTHON), 'eval_rcnn_official.py',
        '--cfg_file', 'cfgs/default.yaml',
        '--ckpt', 'PointRCNN.pth',
        '--batch_size', '1', '--workers', '0',
        '--eval_mode', 'rcnn',
        '--output_dir', str(out_dir / 'inference'),
        '--set', 'RPN.LOC_XZ_FINE', 'False', 'TEST.SPLIT', split_name,
    ]
    (out_dir / 'command.txt').write_text(' '.join(command) + '\n')
    (out_dir / 'checkpoint_used.txt').write_text(str(CKPT) + '\n')
    (out_dir / 'config_used.txt').write_text(str(CFG) + '\n')
    (out_dir / 'split_used.txt').write_text(f'{split_name} ({len(frame_ids(split_file))} frames)\n')

    # NUMBA_ENABLE_CUDASIM=1 avoids a segfault: importing
    # tools/kitti_object_eval_python/evaluate.py triggers a numba @cuda.jit
    # compile at module-import time (rotate_iou.py) that crashes against the
    # real CUDA driver in this environment. The simulator avoids the native
    # driver call. This matches the env var already used in the validated
    # local reference pipeline (results/rpn4096_no_distance_propose_comparison
    # /run_comparison_method.py). Even with the simulator, the in-script
    # Python KITTI AP evaluation can still raise on some inputs (observed on
    # tiny smoke splits) -- that happens strictly after all per-frame
    # detections are written, so it does not affect prediction completeness.
    # We rely on the separate, validated C++ devkit step for AP instead, so a
    # non-zero return code here is only fatal if prediction coverage is
    # incomplete (checked by coverage()).
    env = os.environ.copy()
    env['NUMBA_ENABLE_CUDASIM'] = '1'

    try:
        if VEL.is_symlink() or VEL.exists():
            if VEL.is_dir() and not VEL.is_symlink():
                raise RuntimeError('Refusing to replace non-symlink velodyne directory')
            VEL.unlink(missing_ok=True)
        VEL.symlink_to(input_dir)
        with (out_dir / 'inference_stdout_stderr.log').open('wb') as log:
            start = time.time()
            proc = subprocess.run(command, cwd=TOOLS, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
        (out_dir / 'runtime_seconds.txt').write_text(f'{elapsed:.2f}\n')
        (out_dir / 'returncode.txt').write_text(str(proc.returncode) + '\n')
    finally:
        restore_velodyne(before)
        (out_dir / 'velodyne_after.txt').write_text(record_velodyne(VEL) + '\n')
    return elapsed, proc.returncode


def coverage(out_dir, split_file):
    frames = frame_ids(split_file)
    pred = out_dir / 'inference/eval/epoch_no_number' / Path(split_file).stem / 'final_result/data'
    files = sorted(pred.glob('*.txt')) if pred.exists() else []
    pred_ids = {p.stem for p in files}
    missing = [f for f in frames if f not in pred_ids]
    extra = sorted(pred_ids - set(frames))
    empty = [p.stem for p in files if p.stat().st_size == 0]
    (out_dir / 'empty_prediction_files.txt').write_text('\n'.join(empty) + ('\n' if empty else ''))
    (out_dir / 'prediction_coverage.txt').write_text('\n'.join([
        f'expected={len(frames)}', f'produced={len(files)}', f'missing={len(missing)}',
        f'extra={len(extra)}', f'empty={len(empty)}', f'missing_first50={missing[:50]}', f'extra_first50={extra[:50]}'
    ]) + '\n')
    if missing or extra:
        raise RuntimeError(f'prediction coverage failed: missing={len(missing)} extra={len(extra)}')
    return pred, len(files), len(empty)


def run_cpp_eval(out_dir, pred_dir):
    work = out_dir / 'kitti_cpp_eval_work'
    if work.exists():
        shutil.rmtree(work)
    data_label = work / 'data/object/label_2'
    result_sha = 'eval_result'
    result_data = work / 'results' / result_sha / 'data'
    data_label.mkdir(parents=True)
    result_data.mkdir(parents=True)
    for i in range(7481):
        frame = f'{i:06d}'
        label = data_label / f'{frame}.txt'
        src_label = LABEL_FULL / f'{frame}.txt'
        if src_label.exists():
            label.symlink_to(src_label)
        elif (LABEL_REAL / f'{frame}.txt').exists():
            label.symlink_to(LABEL_REAL / f'{frame}.txt')
        else:
            label.write_text('')
        pred_src = pred_dir / f'{frame}.txt'
        pred_dst = result_data / f'{frame}.txt'
        if pred_src.exists():
            pred_dst.write_bytes(pred_src.read_bytes())
        else:
            pred_dst.write_text('')
    with (out_dir / 'kitti_cpp_eval.log').open('wb') as log:
        proc = subprocess.run([str(CPP_EVAL), result_sha], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'KITTI C++ eval failed with return code {proc.returncode}')
    res = work / 'results' / result_sha
    art = out_dir / 'kitti_cpp_eval_artifacts'
    if art.exists():
        shutil.rmtree(art)
    art.mkdir()
    for fn in ['stats_car_detection.txt', 'stats_car_detection_ground.txt', 'stats_car_detection_3d.txt',
               'stats_car_orientation.txt']:
        src = res / fn
        if src.exists():
            shutil.copy2(src, art / fn)
    for fn in ['car_detection.txt', 'car_detection_ground.txt', 'car_detection_3d.txt']:
        p = res / 'plot' / fn
        if p.exists():
            shutil.copy2(p, art / fn)
    shutil.rmtree(work)
    return art


def ap_from_stats(path):
    """KITTI C++ devkit writes 41 recall points (r=0..1 step 0.025); AP_R40 excludes r=0."""
    if not path.exists():
        return [float('nan')] * 3
    vals = []
    for line in Path(path).read_text().splitlines():
        row = [float(x) for x in line.split()]
        vals.append(sum(row[1:]) / 40.0 * 100.0)
    while len(vals) < 3:
        vals.append(float('nan'))
    return vals[:3]


def write_summary(name, input_dir, out_dir, pred_count, empty_count, elapsed, returncode, art):
    metrics = [
        ('bbox_ap', art / 'stats_car_detection.txt'),
        ('bev_ap', art / 'stats_car_detection_ground.txt'),
        ('3d_ap', art / 'stats_car_detection_3d.txt'),
        ('aos_ap', art / 'stats_car_orientation.txt'),
    ]
    rows = []
    with (out_dir / 'parsed_ap_results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'easy', 'moderate', 'hard'])
        for metric, path in metrics:
            easy, mod, hard = ap_from_stats(path)
            rows.append((metric, easy, mod, hard))
            w.writerow([metric, f'{easy:.6f}', f'{mod:.6f}', f'{hard:.6f}'])
    summary = {
        'name': name,
        'input_dir': str(input_dir),
        'prediction_files': pred_count,
        'empty_prediction_files': empty_count,
        'runtime_seconds': elapsed,
        'inference_returncode': returncode,
        'inference_returncode_note': (
            'non-zero, but tolerated: prediction coverage was complete; '
            'the in-script Python KITTI AP step is not used, only the C++ devkit result below'
        ) if returncode != 0 else None,
        'ap_r40_percent': {m: {'easy': e, 'moderate': mo, 'hard': h} for m, e, mo, h in rows},
        'status': 'PASS',
    }
    (out_dir / 'result_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--split-file', default=str(VAL))
    args = p.parse_args()
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    args.split_file = str(Path(args.split_file).resolve())
    print(f'RUN_START {args.name}', flush=True)
    try:
        elapsed, returncode = run_inference(args.name, input_dir, out_dir, args.split_file)
        pred_dir, pred_count, empty_count = coverage(out_dir, args.split_file)
        if returncode != 0:
            print(f'WARN {args.name} inference exited {returncode} but prediction coverage is complete; continuing', flush=True)
        art = run_cpp_eval(out_dir, pred_dir)
        summary = write_summary(args.name, input_dir, out_dir, pred_count, empty_count, elapsed, returncode, art)
        print(f'RUN_DONE {args.name} predictions={pred_count} empty={empty_count} runtime={elapsed:.1f}s', flush=True)
    except Exception as e:
        (out_dir / 'result_summary.json').write_text(json.dumps({'name': args.name, 'status': 'FAIL', 'error': str(e)}, indent=2))
        print(f'RUN_FAIL {args.name} error={e}', flush=True)
        raise


if __name__ == '__main__':
    main()
