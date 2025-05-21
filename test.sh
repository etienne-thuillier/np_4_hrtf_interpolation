#!/bin/bash

# nodes (see https://scicomp.aalto.fi/triton/tut/gpu/#gpu-computing):

workdir=/scratch/work/thuille1
sessions=$workdir/sessions/np_4_hrtf_interpolation
# resume experiment
experiment=20250421162952

mkdir -p "${sessions}/${experiment}"

# train (full)
python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/models4DynCues/code/train.py paths=triton data=default callbacks=default model=default trainer=default paths.sessions=${sessions} paths.experiment=${experiment}"
## train (tiny)
#python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/models4DynCues/code/train.py paths=triton data=tiny callbacks=tiny model=tiny trainer=tiny paths.sessions=${sessions} paths.experiment=${experiment} model.epochs=50"

## eval (full)
#python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/models4DynCues/code/evaluate_hydra.py --config-path=${sessions}/${experiment}/.hydra --config-name=config.yaml paths.workdir=${sessions}/${experiment} paths.experiment=${experiment} +checkpoint=checkpoint_epoch1040_step43680"

## debug scripts
#python_command="python3 ${workdir}/models4DynCues/code/tmp.py paths=triton data=sofa_dataloader_tiny paths.sessions=${sessions} paths.experiment=${experiment}"
#python_command="HYDRA_FULL_ERROR=1 python3 ${workdir}/models4DynCues/code/tmp.py paths=triton data=sofa_dataloader callbacks=none paths.sessions=${sessions} paths.experiment=${experiment}"

echo ${python_command}

sbatch \
--gres=gpu:1 \
--partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-ellis \
--time=12:00:00 \
--job-name=$job_name \
--mem-per-cpu=24G \
--cpus-per-task=4 \
--output=$sessions/$experiment/slurm.out \
--error=$sessions/$experiment/slurm.err \
--wrap="module load mamba &&
  source activate spherical_neural_process &&
	cd ${workdir}/models4DynCues/code &&
        git show --oneline -s &&
	echo ${python_command} &&
        ${python_command}"
