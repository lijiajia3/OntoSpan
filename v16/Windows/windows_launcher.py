"""Standard-library-only Windows bootstrap, shared by every numbered entry."""
import os
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def child_env():
    return dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')


def local_python():
    return ROOT / '.venv' / 'Scripts' / 'python.exe'


def ensure_runtime():
    # Developer CPU tests can still run directly on macOS/Linux.
    if platform.system() != 'Windows':
        return
    target = local_python()
    if not target.is_file():
        raise SystemExit('本包尚未建立环境。请先用 Python3.11 运行 01_SETUP_WINDOWS.py；不需要手动激活环境。')
    if Path(sys.executable).resolve() == target.resolve():
        if sys.version_info[:2] != (3, 11):
            raise SystemExit('本包 .venv 不是 Python3.11。请保留旧目录，在新目录用 Python3.11 运行01。')
        return
    script = Path(sys.argv[0]).resolve()
    print('切换到本包环境：' + str(target), flush=True)
    raise SystemExit(subprocess.call([str(target), str(script), *sys.argv[1:]],
                                     cwd=ROOT, env=child_env()))


def setup(download=True):
    if platform.system() != 'Windows':
        raise SystemExit('此安装入口用于Windows；不会在当前系统安装CUDA轮子。')
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('请使用Python3.11：py -3.11 01_SETUP_WINDOWS.py')
    target = local_python()
    kwargs = dict(cwd=ROOT, env=child_env(), check=True)
    if not target.is_file():
        subprocess.run([sys.executable, '-m', 'venv', str(ROOT / '.venv')], **kwargs)
    print('安装位置：' + str(target) + '（不修改系统Python或旧v12环境）', flush=True)
    commands = [
        [str(target), '-c', "import sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] == (3, 11) else 'Existing .venv must use Python3.11; use a new folder.')"],
        [str(target), '-m', 'pip', 'install', '--upgrade', 'pip'],
        [str(target), '-m', 'pip', 'install', 'torch==2.9.1', '--index-url', 'https://download.pytorch.org/whl/cu128'],
        [str(target), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements_windows.txt')],
        [str(target), '-c', f'from runner import preflight; preflight(download={download!r})'],
    ]
    for command in commands:
        subprocess.run(command, **kwargs)
    print('01安装和检查完成。接下来运行03_KEY_15.py；02仅检查，不训练。', flush=True)
