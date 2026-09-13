"""Identify the code and runtime behind a measured training step."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


CONTROL_ONLY={'archive.py','budget.py','llm.py','llm_worker.py','providers.py',
              'observe.py','references.py','provenance.py','compile_worker.py'}


def source_files(root):
    root=Path(root)
    paths=[p for folder in ('adapters','kernel_evolution') for p in (root/folder).rglob('*.py')]
    paths += [root/name for name in ('config.py','search.py') if (root/name).exists()]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def identity(root):
    root=Path(root)
    files=source_files(root)
    benchmark={name:digest for name,digest in files.items() if name.startswith('adapters/') or
               (name.startswith('kernel_evolution/') and Path(name).name not in CONTROL_ONLY)}
    versions={'python':sys.version.split()[0]}
    for package in ('torch','triton'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    result={'source_files':files,'runtime_versions':versions,
            'benchmark_digest':hashlib.sha256(json.dumps(benchmark,sort_keys=True).encode()).hexdigest(),
            'git_revision':None,'uploaded_manifest_matches':False}
    manifest=root/'source_manifest.json'
    if manifest.exists():
        uploaded=json.loads(manifest.read_text())
        if uploaded.get('source_files')==files:
            result.update(git_revision=uploaded.get('git_revision'),uploaded_manifest_matches=True)
    if result['git_revision'] is None:
        try:
            process=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,timeout=2)
            if process.returncode==0:result['git_revision']=process.stdout.strip()
        except (OSError,subprocess.TimeoutExpired):pass
    return result


def assert_compatible(previous,current):
    if previous is None:raise ValueError('Prepared run has no source identity; use a fresh run directory')
    for key in ('benchmark_digest','runtime_versions'):
        if previous.get(key)!=current.get(key):
            raise ValueError('Prepared benchmark identity changed: '+key+'; recalibrate in a fresh run directory')
