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

