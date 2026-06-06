Claude & I recently developed [bioimage-cpp](https://github.com/computational-cell-analytics/bioimage-cpp) that bundles efficient image analysis functionality for segmentation
and other tasks in a single C++ library with light weight python bindings. It reimplements functionality from the libraries [affogato](https://github.com/constantinpape/affogato),
[nifty](https://github.com/DerThorsten/nifty), and [vigra](https://github.com/ukoethe/vigra) that underly many of the (segmentation) algorithms and software used and developed by our group.
It also reimplements selected functionality from `scipy.ndimage` and `scikit-image`, such as watersheds and distance transforms, to provide more efficient implementations.

This development had two goals:
- Make the functionality available via pip. `affogato`, `nifty`, and `vigra` have complex dependencies which prevented publishing via PyPI. This will make the installation of our software much easier in the future.
- Improve the efficiency of underyling algorithm (which mostly succeeded, e.g. speeding up watersheds ca. 10x compared to scikit-image's implementation).

A migration guide that explains how to replace the functionality from other libraries can be found [here](https://computational-cell-analytics.github.io/bioimage-cpp/bioimage_cpp.html#migration-guide).
We have now migrated the most important python packages from our group to `bioimage-cpp` and will continue with migrating some more, see details below.

**Important for updating your dependencies:**
- I strongly recommend to create a new conda/mamba environment when installing the software with new dependencies.
- The migration is fairly well tested, but there may be some bugs. If something doesn't work as expected after the migration create an issue in the respective repository.

## Migrated libraries / WIP

The following libraries have already been migrated:
- [elf](https://github.com/constantinpape/elf): version 0.9 is available on conda-forge and PyPI.
- [torch_em](https://github.com/constantinpape/torch-em): version 0.9 is available on conda-forge and PyPI.
- [micro_sam](https://github.com/computational-cell-analytics/micro-sam): version 1.8.1 is available on conda-forge and PyPI.
- [patho-sam](https://github.com/computational-cell-analytics/patho-sam): soon
- [medico-sam](https://github.com/computational-cell-analytics/medico-sam): soon
- [peft-sam](https://github.com/computational-cell-analytics/peft-sam): soon

Migration for the following libraries is either under way or will happen soon:
- [synapse-net](https://github.com/computational-cell-analytics/synapse-net): see https://github.com/computational-cell-analytics/synapse-net/pull/167

## Out-of-scope / pending

The following libraries will not yet be migrated:
- [cluster_tools](https://github.com/constantinpape/cluster_tools)
- [coclhea-net](https://github.com/computational-cell-analytics/cochlea-net)
- [mobie-utils](https://github.com/mobie/mobie-utils-python)

They require a migration of `cluster_tools`, which requires on `nifty`-functionality that has not yet been migrated because it relies
on file I/O (from zarr/n5 files) in C++. Bioimage-cpp currently does not support this to avoid dependencies that make it difficult to distribute via pip.
We will eventually update [z5](https://github.com/constantinpape/z5) so that it can be published on PyPI and then use this for I/O in C++.
This will enable to continue with the migration (or, in the case of `cluster_tools` to implement a more modern alternative for distributed image analysis).

In the meantime, the dependencies of these packages are pinned to versions of `elf` and `torch-em` that still use the "old" stack based on `nifty` etc.
