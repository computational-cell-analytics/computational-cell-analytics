# Running [`napari`](https://napari.org/stable/usage.html) on SCC using X2Go:

Important points:
- Install the environment from your local terminal/powershell first, later access them on the VM (installing modules via VM (sometimes) creates strange conflicts/issues)
- Use Jupyter/Vim (via terminal) to edit your python scripts on the VM
- (for Jupyter users) You need to open firefox from the terminal itself (it needs to be installed using `mamba install firefox`)
- napari is a little slow on the front nodes right now (a little faster on GPU nodes though), ignore the delays for now.

## Getting X2GO GUI:
- Follow the link in our wiki [here](https://github.com/computational-cell-analytics/computational-cell-analytics/wiki/Using-the-university-cluster#gui--jupyter)

## Getting a test environment to check napari on VM:
- Get your environment using the [yaml](https://github.com/computational-cell-analytics/computational-cell-analytics/blob/main/napari-scc/environment_napari.yaml) file
> Note: Before adding other modules, first cross-check using the basic dependencies work or not (for napari)

## Script for getting Interactive GPU nodes working with napari:
- `srun --x11 --pty -p gpu -G gtx1080:1 -t 12:00:00 /bin/bash` (would get you a GPU job, from where you can now forward "napari" requests and should work just fine)
- `napari` - should open an empty console for you to test
