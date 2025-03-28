# OpenHelix: An Open-source Dual-System VLA Model for Robotics Manipulation
By [Can Cui*](https://cuixxx.github.io) and [Pengxiang Ding*](https://dingpx.github.io)

This is our re-implementation of the [Helix](https://www.figure.ai/news/helix).

# Model overview and stand-alone usage
To facilitate fast development on top of our model, we provide here an [overview of our implementation of 3D Diffuser Actor](./docs/OVERVIEW.md).

The model can be indenpendently installed and used as stand-alone package.
```
> pip install -e .
# import the model
> from diffuser_actor import DiffuserActor, Act3D
> model = DiffuserActor(...)
```

# Installation
Create a conda environment with the following command:

```
# initiate conda env
> conda update conda
> conda env create -f environment.yaml
> conda activate 3d_diffuser_actor

# install diffuser
> pip install diffusers["torch"]

# install dgl (https://www.dgl.ai/pages/start.html)
>  pip install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl

# install flash attention (https://github.com/Dao-AILab/flash-attention#installation-and-features)
> pip install packaging
> pip install ninja
> pip install flash-attn --no-build-isolation
```

### Install CALVIN locally

Remember to use the latest `calvin_env` module, which fixes bugs of `turn_off_led`.  See this [post](https://github.com/mees/calvin/issues/32#issuecomment-1363352121) for detail.
```
> git clone --recurse-submodules https://github.com/mees/calvin.git
> export CALVIN_ROOT=$(pwd)/calvin
> cd calvin
> cd calvin_env; git checkout main
> cd ..
> ./install.sh; cd ..
```

# Data Preparation

See [Preparing CALVIN dataset](./docs/DATA_PREPARATION_CALVIN.md).


### (Optional) Encode language instructions

We provide our scripts for encoding language instructions with CLIP Text Encoder on CALVIN.  Otherwise, you can find the encoded instructions on CALVIN and RLBench ([Link](https://huggingface.co/katefgroup/3d_diffuser_actor/blob/main/instructions.zip)).
```
> python data_preprocessing/preprocess_calvin_instructions.py --output instructions/calvin_task_ABC_D/validation.pkl --model_max_length 16 --annotation_path ./calvin/dataset/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy

> python data_preprocessing/preprocess_calvin_instructions.py --output instructions/calvin_task_ABC_D/training.pkl --model_max_length 16 --annotation_path ./calvin/dataset/task_ABC_D/training/lang_annotations/auto_lang_ann.npy
```

# Getting started

See [Getting started with CALVIN](./docs/GETTING_STARTED_CALVIN.md).


# Acknowledgement
Parts of this codebase have been adapted from [Act3D](https://github.com/zhouxian/act3d-chained-diffuser) and [CALVIN](https://github.com/mees/calvin).
