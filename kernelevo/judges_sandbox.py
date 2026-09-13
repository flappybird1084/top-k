"""GPU runner helpers with per-run Unix identities and a clean environment."""
import glob
import os
import shutil


def user_command(work, argv, uid):
    """Separate Unix identity for each run on a containerized GPU host.

    The host may deny nested bind mounts. This isolates writable run files and
    process credentials, but is not a replacement for a VM security boundary.
    """
    if not shutil.which('setpriv') or uid < 200000:
        raise RuntimeError('Per-run process isolation is unavailable')
    for root, dirs, files in os.walk(work):
        os.chown(root, uid, uid)
        for name in files:
            os.chown(os.path.join(root, name), uid, uid, follow_symlinks=False)
    os.chmod(work, 0o700)
    home = work + '/home'
    os.makedirs(home, exist_ok=True)
    os.chown(home, uid, uid)
    env = {'HOME': home, 'USER': 'runner', 'LOGNAME': 'runner', 'PATH': '/usr/local/bin:/usr/bin:/bin',
           'PYTHONUNBUFFERED': '1', 'TMPDIR': home,
           'KEVO_RELAY_DIR': work + '/run/search_relay', 'WANDB_MODE': 'disabled'}
    return ['/usr/bin/env', '-i', *[k+'='+v for k, v in env.items()],
            '/usr/bin/setpriv', '--reuid='+str(uid), '--regid='+str(uid),
            '--clear-groups', '--no-new-privs', '--bounding-set=-all', *argv]


def command(work, argv, environment):
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise RuntimeError('GPU isolation is unavailable; refusing to execute a public repository')
    args = [bwrap, '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
            '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv']
    for path in ('/usr', '/bin', '/sbin', '/lib', '/lib64', '/opt/cuda'):
        if os.path.exists(path):
            args += ['--ro-bind', path, path]
    for path in ('/etc/ssl', '/etc/resolv.conf', '/etc/hosts', '/etc/ld.so.cache'):
        if os.path.exists(path):
            args += ['--ro-bind', path, path]
    args += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/home/runner']
    for path in glob.glob('/dev/nvidia*'):
        args += ['--dev-bind', path, path]
    args += ['--bind', work, work, '--chdir', work]
    clean = {'HOME': '/home/runner', 'PATH': '/usr/local/bin:/usr/bin:/bin',
             'PYTHONUNBUFFERED': '1', 'CUDA_CACHE_PATH': '/tmp/cuda-cache',
             'KEVO_RELAY_DIR': work + '/run/search_relay', 'WANDB_MODE': 'disabled'}
    # No account credentials or notebook authentication enter this sandbox.
    for key in ('LD_LIBRARY_PATH', 'CUDA_HOME'):
        if os.environ.get(key):
            clean[key] = os.environ[key]
    for key, value in clean.items():
        args += ['--setenv', key, str(value)]
    return args + ['--'] + argv
