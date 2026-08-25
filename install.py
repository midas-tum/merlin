import os
import sys
import shutil
import subprocess
import pathlib

cwd = pathlib.Path().absolute()
py_cmd = sys.executable or 'python3'

def get_install_cmd():
    # Check if uv is available or try running via python -m uv
    try:
        res = subprocess.run(['uv', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            return ['uv', 'pip', 'install', '-e', '.']
    except Exception:
        pass

    try:
        res = subprocess.run([py_cmd, '-m', 'uv', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            return [py_cmd, '-m', 'uv', 'pip', 'install', '-e', '.']
    except Exception:
        pass

    return [py_cmd, '-m', 'pip', 'install', '-e', '.']

inst_cmd = get_install_cmd()

os.chdir('python')
try:
    res = subprocess.run([py_cmd, 'setup.py', 'sampling'])
    if res.returncode != 0:
        subprocess.run(inst_cmd, check=True)
except Exception:
    subprocess.run(inst_cmd, check=True)

os.chdir(str(cwd))
os.chdir('pytorch')
subprocess.run(inst_cmd, check=True)

os.chdir(str(cwd))
os.chdir('tensorflow')
subprocess.run(inst_cmd, check=True)

os.chdir(str(cwd))