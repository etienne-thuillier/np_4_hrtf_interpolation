# HRTF Interpolation Using a Spherical Neural Process Meta-Learner

This repository contains the official code for our paper:

> [**HRTF Interpolation Using a Spherical Neural Process Meta-Learner**](https://ieeexplore.ieee.org/document/10418851)  
> Etienne Thuillier, Craig T. Jin, Vesa Välimäki  
> *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024*

It includes the **Spherical Convolutional Conditional Neural Process** (SConvCNP) model introduced in the paper
and the **barycentric interpolation** baseline.

## Install

### Source code

Clone the *spin_spherical_cnns* part of the *google-research* repository:
```
mkdir google-research
cd google-research
git init
git remote add origin https://github.com/google-research/google-research.git
git sparse-checkout init
git sparse-checkout set spin_spherical_cnns
git fetch --depth=1 origin a32a932640d0b83a3523917370c383cad14839fc
git checkout -b master FETCH_HEAD
cd ..
```

Clone *np_4_hrtf_interpolation* (this repo):
```
git clone https://github.com/etienne-thuillier/np_4_hrtf_interpolation.git
```

Link *spin_spherical_cnns* as a subfolder into *np_4_hrtf_interpolation*:
```
cd np_4_hrtf_interpolation
ln -s ../google-research/spin_spherical_cnns spin_spherical_cnns
```

### Dataset

Download simulated HRTF sets from the HUTUBS dataset:
```
mkdir <your dataset directory here>/sofa
cd <your dataset directory here>/sofa
wget -r -np -nH --cut-dirs=2 -A "*_simulated.sofa" --wait=2 --retry-connrefused --tries=10 -c https://sofacoustics.org/data/database/hutubs/
``` 
Check you have all required files.
Running the following command should return 99.
```
find <your dataset directory here>/sofa/hutubs -name "*_simulated.sofa" | wc -l
```

### Paths

Edit the following fields of ```conf/paths/default.yaml```
```
data: <your dataset directory here>
sessions: <your sessions directory here>
```

### Environment
```
conda env create -f environments/spherical_neural_process.yaml
```

## Train
Command-line calls are customisable following [Hydra](https://hydra.cc) syntax.
Configuration files are located in the ```conf``` folder.

To train with the SConvCNP model under the published (i.e. default) configuration, run 
```
python3 train.py
```
This will create a new experiment folder in ```<your sessions directory here>``` where will model weights and tensorboard data will be saved.

Note that training the published model requires 62 GB of GPU memory.

To step through code while running a small model, use
```
python3 train.py trainer=tiny data=tiny callbacks=tiny model=tiny
```

To improve debugging using non-jit-compiled code, run
```
python3 train.py trainer=tiny data=tiny callbacks=tiny model=tiny jax=debug
```

## Evaluate (! Under Construction)

### SConvCNP

Download the pre-trained weights ***** under construction!
```
cd <your sessions directory here> 
wget http://research.spa.aalto.fi/publications/papers/ieee-taslp-2024-hrtf-interp/*********
```
Produce tensorboard output for test set
```
python3 test.py --config-path=<your sessions directory here>/*********/.hydra --config-name=config.yaml paths.workdir=<your sessions directory here>/********* paths.experiment=********* +checkpoint=checkpoint_epoch4_step8 
```
Inspect the result
```
tensorboard --logdir=<your sessions directory here>/*********
```

### Barycentric interpolation (baseline)
```
python3 test.py model=barycentric jax.jax_disable_jit=true +checkpoint=null
```
This will create a new experiment folder in ```<your sessions directory here>```.

Note that the implementation of this baseline was carried-out with accuracy in mind, not efficiency.
The approach is suboptimal in terms of computational complexity.


## License

This project is licensed under the MIT License.

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{thuillier2024hrtf,
  author={Thuillier, Etienne and Jin, Craig T. and Välimäki, Vesa},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={HRTF Interpolation Using a Spherical Neural Process Meta-Learner},
  year={2024},
  volume={32},
  pages={1790--1802},
  doi={10.1109/TASLP.2024.3361377}
}
