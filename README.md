# Special-Train
## Deployment using lightning for getting frames and segmentation.

### Cone repository

```
git clone https://github.com/karkisa/special-train.git
```

### Activate Environment
```
conda actovate environemt_name
```

make the repo folder the base folder in terminal

```
cd special-train
```

```
pip install -r requirements.txt
```
### launch DeteX

```
lightning run app app_detection.py
```

### launch XtraX

```
lightning run app app_sam.py
```
