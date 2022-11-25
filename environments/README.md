# Base environment file

This directory contains different environment files to reproduce commonly used conda environments.

An environment can be created by running
```
conda env create -f <ENV_FILE> -n <ENV_NAME>
```
For example
```
cnda env create -f main_env.yaml -n my-environment
```

If `mamba` is installed you can also run `mamba env create ...`.

Currently available environments:
- `main_env.yaml`: this was used to create the `main310` environment on the SCC cluster. Note that `torch_em` is not installed as part of the environment dependency, but is instead installed in development mode (as described [here](https://github.com/constantinpape/torch-em#from-source))
