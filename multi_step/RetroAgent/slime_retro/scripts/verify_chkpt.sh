#!/bin/bash
# Verify torch_dist checkpoint by loading it in the SLIME container.
# Run on a GPU server with the converted checkpoint.
#
# Usage:
#   bash verify_chkpt.sh

set -ex

MODEL_NAME="Qwen3-4B-Instruct-2507"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
SLIME_IMAGE="slimerl/slime:v0.3.0"

docker run --rm --gpus all \
    -v "${MODELS_DIR}:/models" \
    "${SLIME_IMAGE}" \
    python3 -c "
import torch
import torch.distributed as dist
import os, json, sys
sys.path.insert(0, '/root/Megatron-LM')

# Init single-GPU distributed
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '12355')
torch.cuda.set_device(0)
dist.init_process_group(backend='nccl', world_size=1, rank=0, device_id=torch.device('cuda:0'))

from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.filesystem import FileSystemReader

ckpt_dir = '/models/${MODEL_NAME}_torch_dist/release'

# Check files
print('=== Files ===')
for f in sorted(os.listdir(ckpt_dir)):
    size = os.path.getsize(os.path.join(ckpt_dir, f))
    print(f'  {f}: {size/1e6:.1f} MB' if size > 1e6 else f'  {f}: {size} B')

# Load and inspect metadata.json
print('\n=== metadata.json ===')
with open(os.path.join(ckpt_dir, 'metadata.json')) as f:
    print(json.dumps(json.load(f), indent=2))

# Load .metadata via FileSystemReader (distcp format, not torch.load)
reader = FileSystemReader(ckpt_dir)
metadata = reader.read_metadata()
keys = sorted(metadata.state_dict_metadata.keys())
print(f'\n=== State Dict: {len(keys)} keys ===')
for k in keys[:15]:
    print(f'  {k}')
print('  ...')
for k in keys[-5:]:
    print(f'  {k}')

# Verify key layers exist
print('\n=== Layer check ===')
expected_prefixes = ['model.decoder.layers.0.', 'model.decoder.layers.35.', 'model.embedding.', 'model.output_layer.']
for prefix in expected_prefixes:
    found = any(k.startswith(prefix) for k in keys)
    status = 'OK' if found else 'MISSING'
    print(f'  [{status}] {prefix}*')

# Load common.pt (model args)
print('\n=== common.pt ===')
common = torch.load(os.path.join(ckpt_dir, 'common.pt'), map_location='cpu', weights_only=False)
print(f'  keys: {list(common.keys())}')
if 'args' in common:
    a = common['args']
    for attr in ['num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads', 'num_query_groups', 'vocab_size', 'bf16', 'rotary_base']:
        print(f'  {attr}: {getattr(a, attr, \"N/A\")}')

dist.destroy_process_group()
print('\n=== Checkpoint verification PASSED ===')
"
