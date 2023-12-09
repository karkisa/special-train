# Special-Train
## Deployment using lightning for getting frames and segmentation.

### Cone repository

```
git clone https://github.com/karkisa/special-train.git
```
### Create Environment
Lets create an environment "Wahel_Morph" (You are free to use any other name for your environment)

```
conda create --name Whale_Morph python=3.8

```
Use the environment.yaml file to get all teh packages need for the application

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

The files saved by DeteX follows the format: 
VideoName_HH_MM_SS_HM_HC.png 

“VideoName” is the name of the video from which it was selected. HH_MM_SS is the time stamp where HH refers to hours, MM refers to minutes and SS refers to seconds. HM_HC captures altitude information. For example, if the height is. 45.6 then HM is 45 and HC is 6.  

Follow this video tutorial for DeteX application use

### launch XtraX

```
lightning run app app_sam.py
```
follow this video tutorial for XtraX application use
