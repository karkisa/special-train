# Special-Train
## Deployment using lightning for getting frames and segmentation.

### Cone repository

```
git clone https://github.com/karkisa/special-train.git
```
### Create Environment
Lets create an environment "Wahel_Morph" (You are free to use any other name for your environment)

```
conda create --name Whale_Morph pytohn=3.8

```Use the environment.yaml file to get all teh packages need for the application

```
conda env update --name Whale_Morph --file environment.yml

```

### Activate Environment
```
conda activate Whale_Morph
```

make the repo folder the base folder in terminal

```
cd special-train
```

### launch DeteX

```
lightning run app app_detection.py
```

### launch XtraX

```
lightning run app app_sam.py
```
