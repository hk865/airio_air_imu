"""Strictly load a historical checkpoint and run a small synthetic CPU inference."""
import argparse
import json
import sys
from pathlib import Path

import torch
from pyhocon import ConfigFactory


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from tensors(nested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', choices=['airimu', 'airio'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = root / args.project
    sys.path.insert(0, str(project))
    from model import net_dict
    conf = ConfigFactory.parse_file(str(project / args.config)).train
    conf['precision'] = 'float32'
    conf['device'] = 'cpu'
    conf['sampling'] = False
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = net_dict[conf.network](conf).float().cpu().eval()
    checkpoint = torch.load(root / args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert all(torch.isfinite(t).all() for t in tensors(checkpoint))
    data = {'acc': torch.randn(1, 1009, 3), 'gyro': torch.randn(1, 1009, 3) * 0.01}
    with torch.no_grad():
        output = model.inference(data) if args.project == 'airimu' else model(data, torch.zeros(1, 1009, 3))
    values = list(tensors(output))
    assert values and all(t.numel() and torch.isfinite(t).all() for t in values)
    print(json.dumps({'checkpoint': args.checkpoint, 'epoch': checkpoint['epoch'],
                      'network': conf.network, 'strict_load': True,
                      'synthetic_cpu_inference': 'passed', 'output_shapes': [list(t.shape) for t in values]}))


if __name__ == '__main__':
    main()
