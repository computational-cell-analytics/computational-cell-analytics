#! /home/uni02/UMIN/pape41/Work/software/conda/mambaforge/bin/python
# NOTE: replace this with the path to your conda installation

import os
import sys
import inspect
import subprocess
from datetime import datetime

# two hours / days in minutes
TWO_HOURS = 2 * 60
TWO_DAYS = 2 * 24 * 60

# currently available gpu types on SCC
GPU_TYPES = ["gtx980", "gtx1080", "k40"]


def write_batch_script(script, out_path, env_name,
                       n_threads, gpu_type, n_gpus,
                       mem_limit, time_limit, exclude_nodes):
    batch_script = f"""#! /bin/bash
#SBATCH -N 1
#SBATCH -c {n_threads}
#SBATCH --mem {mem_limit}
#SBATCH -t {time_limit}
#SBATCH -p gpu
#SBATCH -G {gpu_type}:{n_gpus}
"""
    # set qos depending on the runtime
    if time_limit < TWO_HOURS:
        batch_script += f"#SBATCH --qos short\n"
    if time_limit > TWO_DAYS:
        batch_script += f"#SBATCH --qos long\n"

    if exclude_nodes is not None:
        batch_script += "#SBATCH --exclude={','.join(exclude_nodes)}\n"

    batch_script += f"""
source ~/.bashrc
conda activate {env_name}
python {script} $@
"""
    with open(out_path, 'w') as f:
        f.write(batch_script)


def submit_slurm(script, input_, n_threads=7, n_gpus=1,
                 gpu_type="gtx1080", mem_limit="64G",
                 time_limit=TWO_DAYS, env_name=None, exclude_nodes=None):
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    if gpu_type not in GPU_TYPES:
        raise ValueError(f"Invalid gpu type, only the types {GPU_TYPES} are available.")

    tmp_folder = os.path.expanduser("~/.gpu_jobs")
    os.makedirs(tmp_folder, exist_ok=True)

    print("Submitting training script", script)
    print("with arguments", " ".join(input_))

    script_name = os.path.split(script)[1]
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = os.path.splitext(script_name)[0] + dt

    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    log = os.path.join(tmp_folder, f"{tmp_name}.log")
    err = os.path.join(tmp_folder, f"{tmp_name}.err")
    print("The job files are stored here:")
    print("Batch script:", batch_script)
    print("Log file:    ", log)
    print("Error log:   ", err)

    if env_name is None:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", None)
        if env_name is None:
            raise RuntimeError("Could not find conda")

    write_batch_script(script, batch_script, env_name,
                       int(n_threads), gpu_type, int(n_gpus),
                       mem_limit, int(time_limit),
                       exclude_nodes=exclude_nodes)

    cmd = ["sbatch", "-o", log, "-e", err, "-J", script_name, batch_script]
    cmd.extend(input_)
    subprocess.run(cmd)


def scrape_kwargs(input_):
    params = inspect.signature(submit_slurm).parameters
    kwarg_names = [name for name in params
                   if params[name].default != inspect._empty]
    kwarg_positions = [i for i, inp in enumerate(input_)
                       if inp in kwarg_names]

    kwargs = {input_[i]: input_[i + 1] for i in kwarg_positions}

    kwarg_positions += [i + 1 for i in kwarg_positions]
    input_ = [inp for i, inp in enumerate(input_) if i not in kwarg_positions]

    return input_, kwargs


if __name__ == '__main__':
    script = os.path.realpath(os.path.abspath(sys.argv[1]))
    input_ = sys.argv[2:]

    # scrape the additional arguments (n_threads, mem_limit, etc. from the input)
    input_, kwargs = scrape_kwargs(input_)
    submit_slurm(script, input_, **kwargs)
