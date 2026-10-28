import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_identity(root, data_root, config, checkpoint, spec, options, samples,
                   m2f_config=None):
    root = Path(root).resolve(); data_root = Path(data_root).resolve()
    checkpoint_keys = ('sha256', 'config_sha256')
    models = {'eomt': {k: checkpoint[k] for k in checkpoint_keys if k in checkpoint}}
    if 'm2f' in checkpoint:
        models['m2f'] = {k: checkpoint['m2f'][k] for k in checkpoint_keys if k in checkpoint['m2f']}
    configs = {'eomt': file_sha(config)}
    if m2f_config:
        configs['m2f'] = file_sha(m2f_config)
    sources = ['datasets/sea_panoptic.py', 'tools/analyze_panoptic_errors.py',
               'tools/reobs_core.py', 'tools/eomt_infer.py',
               'tools/tiled_panoptic_infer.py', 'training/lightning_module.py']
    if spec['family'] == 'm2f':
        sources.append('tools/m2f_backend.py')
    source_hashes = {name: file_sha(root / name) for name in sources}
    for name in ('run_eval.py', 'evaluation_identity.py'):
        source_hashes['evaluation/' + name] = file_sha(Path(__file__).parent / name)
    versions = {}
    for package in ('torch', 'torchvision', 'torchmetrics', 'lightning', 'timm',
                    'mmdet', 'mmcv', 'mmengine'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(schema_version=1, spec=spec, effective_options=options,
                checkpoints=models, config_sha256=configs,
                data=dict(root=str(data_root), split=spec['split'],
                          annotations_sha256=file_sha(data_root / 'annotations' / ('panoptic_' + spec['split'] + '.json')),
                          split_images=len(samples),
                          sample_order_sha256=fingerprint([[str(x[0]), str(x[1])] for x in samples])),
                source_sha256=source_hashes, runtime=dict(python=sys.version, packages=versions),
                native_statistics_version=1)


def validate_shard_identities(parts, spec):
    if not parts:
        raise ValueError('no evaluation shards')
    common = None; common_fingerprint = None
    for part in parts:
        identity = part.get('evaluation_identity')
        claimed = part.get('evaluation_fingerprint')
        if not isinstance(identity, dict) or not claimed:
            raise ValueError('missing evaluation identity; legacy shards must be rerun')
        if identity.get('schema_version') != 1 or fingerprint(identity) != claimed:
            raise ValueError('corrupt evaluation fingerprint')
        for key in ('id', 'family', 'split', 'selector', 'window', 'k'):
            if identity['spec'].get(key) != spec.get(key):
                raise ValueError('wrong logical evaluation spec: ' + key)
        if part.get('split_images') != identity['data']['split_images']:
            raise ValueError('shard split size differs from its evaluation identity')
        if common is None:
            common = identity; common_fingerprint = claimed
        elif claimed != common_fingerprint or identity != common:
            raise ValueError('evaluation shard identities differ (checkpoint, data, configuration, selection, source or runtime)')
    return common, common_fingerprint
