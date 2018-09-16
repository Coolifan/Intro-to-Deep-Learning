This is a singularity configuration file for a tensorflow-gpu version in python 3.6.  This should only be used for builds explicitly using the GPU.
Much thanks to Ricardo Henao for the original file.

To build the image:
sudo singularity build tfgpu.simg Singularity.def

If a writable image is desired:
sudo singularity build --writable tfgpu.simg Singularity.def

To go into a shell inside:
singularity shell --nv tfgpu.simg
Note that "--nv" is necessary to load the nvidia drivers.

Jupyter Notebook is installed.  However, there have been permissions issues.
For me, this was solved by running:
`export XDG_RUNTIME_DIR=''`
prior to calling jupyter notebook.