"""Verify the complete delivered file set against SHA256SUMS."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
count = 0
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, relative = line.split('  ', 1)
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    count += 1
print(f'{count} file hashes verified')
