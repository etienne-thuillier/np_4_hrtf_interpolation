#!/bin/bash

# nodes (see https://scicomp.aalto.fi/triton/tut/gpu/#gpu-computing):

workdir=/scratch/work/thuille1
sessions=$workdir/sessions/np_4_hrtf_interpolation
experiment=$(date +"%Y%m%d%H%M%S")

mkdir -p "${sessions}/${experiment}"

## barycentric
#python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/np_4_hrtf_interpolation/test.py paths=triton trainer=default data=default callbacks=default model=barycentric jax.jax_disable_jit=true paths.sessions=${sessions} paths.experiment=${experiment} +checkpoint=null"

# barycentric (non-centered features)
python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/np_4_hrtf_interpolation/test.py paths=triton trainer=default data=default callbacks=default model=barycentric model.input_preprocessing=identity jax.jax_disable_jit=true paths.sessions=${sessions} paths.experiment=${experiment} +checkpoint=null"

## spherical spline
#python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/np_4_hrtf_interpolation/test.py paths=triton trainer=default data=default callbacks=default model=spherical_spline jax.jax_disable_jit=true paths.sessions=${sessions} paths.experiment=${experiment} +checkpoint=null"

## spherical spline
#python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/np_4_hrtf_interpolation/test.py paths=triton trainer=default data=default callbacks=default model=spherical_spline model.input_preprocessing=identity jax.jax_disable_jit=true paths.sessions=${sessions} paths.experiment=${experiment} +checkpoint=null"

echo ${python_command}

sbatch \
--time=02:00:00 \
--job-name=$job_name \
--mem-per-cpu=24G \
--cpus-per-task=4 \
--export=JAX_PLATFORMS=cpu \
--output=$sessions/$experiment/slurm.out \
--error=$sessions/$experiment/slurm.err \
--wrap="module load mamba &&
  source activate spherical_neural_process &&
	cd ${workdir}/models4DynCues/code &&
        git show --oneline -s &&
	echo ${python_command} &&
        ${python_command}"
