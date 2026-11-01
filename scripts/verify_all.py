"""Run bounded CPU checks; never starts training or downloads data."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled', PYTHONPATH='.')
cases = [
    ('airimu', 'configs/exp/EuRoC/codenet.conf', 'experiments/EuRoC/codenet/ckpt/best_model.ckpt'),
    ('airimu', 'configs/exp/EuRoC/freqnet_phase_aware.conf', 'experiments/EuRoC/freqnet_phase_aware/ckpt/best_model.ckpt'),
    ('airimu', 'configs/exp/EuRoC/freqnet_no_phase.conf', 'experiments/EuRoC/freqnet_no_phase/ckpt/best_model.ckpt'),
    ('airio', 'configs/EuRoC/motion_body_rot.conf', 'experiments/euroc/motion_body_rot/ckpt/best_model.ckpt'),
    ('airio', 'configs/EuRoC/motion_body_freq_medium_rot.conf', 'experiments/euroc/motion_body_freq_medium_rot/ckpt/best_model.ckpt'),
]
tests = []
for project in ['airimu', 'airio']:
    cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests']
    result = subprocess.run(cmd, cwd=root/project, env=env, capture_output=True, text=True)
    summary = next((x for x in reversed(result.stdout.splitlines()) if 'passed' in x or 'failed' in x), '')
    tests.append({'project': project, 'command': 'python -m pytest -q -p no:cacheprovider tests',
                  'exit_code': result.returncode, 'summary': summary})
    print(project, summary)
    if result.returncode:
        print(result.stdout, result.stderr)

checkpoints = []
for project, config, checkpoint in cases:
    relative = project + '/' + checkpoint
    cmd = [sys.executable, 'scripts/verify_model.py', '--project', project,
           '--config', config, '--checkpoint', relative]
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    item = {'checkpoint': relative, 'config': project+'/'+config, 'bytes': (root/relative).stat().st_size,
            'sha256': hashlib.sha256((root/relative).read_bytes()).hexdigest(), 'exit_code': result.returncode}
    if result.returncode:
        item['error'] = result.stderr.splitlines()[-1]
        print(result.stderr)
    else:
        item.update(json.loads(result.stdout.splitlines()[-1]))
    checkpoints.append(item)
    print(relative, 'passed' if result.returncode == 0 else 'FAILED')

result = subprocess.run([sys.executable, 'scripts/verify_outputs.py'], cwd=root, env=env, capture_output=True, text=True)
predictions = json.loads(result.stdout) if result.returncode == 0 else {'error': result.stderr.splitlines()[-1]}
for name, content in [('unit_tests', tests), ('checkpoint_validation', checkpoints), ('prediction_validation', predictions)]:
    (root/'reports'/f'{name}.json').write_text(json.dumps(content, indent=2)+'\n')
assert all(x['exit_code'] == 0 for x in tests + checkpoints) and result.returncode == 0
print('CPU verification completed; inspect prediction_validation.json for historical warnings.')
