# Environment files

This directory contains conda environment files to reproduce commonly used environments.

An environment can be created by running
```
mamba env create -f <ENV_FILE> -n <ENV_NAME>
```
For example
```
mamba env create -f main_env.yaml -n main
```

If `mamba` is not installed you can run the same commands with `conda`.

## Currently available environments

- `main_env.yaml`: An environment with the dependencies for deep learning based pytorch projects. Note that [torch_em](https://github.com/constantinpape/torch-em#from-source) (a pytorch based library for deep learning applied to microscopy that I maintain) is not installed as part of the environment. It should instead be installed in development mode to enable updating it. You can do this via the following steps (after you have installed the conda environment):
    - Activate your environment (assuming it's called main): `conda activate main`
    - Clone the torch_em repository: `git clone https://github.com/constantinpape/torch-em`
    - Enter the repository: `cd torch-em`
    - Install it in dev mode: `pip install -e .`
