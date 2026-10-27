# AnatoVL-MT

This repository contains the grade-label split and the scripts used for the reported runs.

`grade_label_split.csv` has one row per image. The columns are `dataset`, `image_id`, `split`, `grade`, and `vertical_cdr`. The grade is a four-level morphological stratum from the vertical cup-to-disc ratio. It is not a glaucoma diagnosis. Drishti-GS1 is split 71/10/20 and RIM-ONE is split 249/36/37.

Fundus images are not included. Obtain Drishti-GS1 and RIM-ONE from their original providers.

The Python files under `medclipseg_cls_patch/` are modifications of [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg). They are not a standalone installation. Place them in a MedCLIPSeg checkout as described in `train_joint.py`. The files under `baselines/` train the U-Net, CLIPSeg, and MaPLe-style joint baselines.

Prompt, freeze, and loss ablation scripts are included. Their saved classification reports used 77 cup images and 38 disc images. They were not rescored on the 37-image RIM-ONE test list in `grade_label_split.csv`.
