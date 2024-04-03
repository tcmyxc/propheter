Propheter: Prophetic Teacher Guided Long-Tailed Distribution Learning
========

![conference](https://img.shields.io/badge/Conference-ICONIP_2023-ff69b4)
&emsp;![code](https://img.shields.io/badge/Code-Official-brightgreen)
&emsp;![doc](https://img.shields.io/badge/Docs-Latest-orange)

> Paper Link: [[arXiv]](https://arxiv.org/abs/2304.04135)

# Operation Manual

> We only provide the results reproduction process of the `cifar-10-lt-ir100` dataset on the `ResNet32` model.
>
> You can get long-tail datasets with different balance factors by modifying file `preprocessing/imbalance_cifar.py`.

## 1.Software versions

The main software versions are as follows:

- Python: 3.6.13

- torch: 1.10.2

- torchvision: 0.11.3

Other dependencies are in the file `requirements.txt`

## 2.Create some directories

```bash
cd propheter
mkdir data
mkdir work_dir
```

## 3.Make dataset

```python
python3 preprocessing/imbalance_cifar.py
```

You will get the `cifar-10-lt-ir100` dataset in the `data` directory.

## 4.Get baseline

```bash
bash train_baseline.sh
```

## 5.Our method

### 5.1 One stage

```bash
bash one_stage.sh
```

The resulting weight file will serve as the weight for the teacher model in the second stage.

### 5.2 Two stage

Replace the value of the variable `cifar_10_lt_ir100_bsl_teacher_model_path` in file `two_stage.sh` with the weights obtained in the first stage. Then:

```bash
bash two_stage.sh
```

## 6.Parameter Settings

We utilize the SGD optimizer with a momentum of 0.9 and a weight decay of 5e-4, 
with a batch size of 128 and a cosine learning rate schedule for the CIFAR10/CIFAR100-LT datasets. 
The initial learning rate is set to 0.01. We employ AutoAugment and Cutout as the data augmentation strategies 
for classification when using CE Loss, Focal Loss, and Balanced Softmax Loss. 

For the ImageNet-LT dataset, we train residual networks with a batch size of 64, a learning rate of 0.025, 
an SGD optimizer with momentum 0.9, and weight decay 1e-4. 

For the Places-LT dataset, we use batch size 64, cosine learning rate schedule decaying from 0.005 to 0. 

# Citation

If you find the paper useful in your research, please consider citing:

```bibtex
@inproceedings{xu2023propheter,
    title={Propheter: Prophetic Teacher Guided Long-Tailed Distribution Learning},
    author={Xu, Wenxiang and Jing, Yongcheng and Zhou, Linyun and Huang, Wenqi and Cheng, Lechao and Feng, Zunlei and Song, Mingli},
    booktitle={ICONIP},
    year={2023},
}
```

