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

make the repo folder the base folder in terminal

```
cd special-train
```

Use the environment.yaml file to get all teh packages need for the application

```
conda env update --name Whale_Morph --file environment.yml

```

### Activate Environment
```
conda activate Whale_Morph
```


### download data
[Yolov5 model weights needed for Detex](https://oregonstate.box.com/s/4bl2pr0xuygbai8gu97hajjs0ihprc7w)
[SAM model weights neede for extrax](https://oregonstate.box.com/s/oltsl30mxvmqvsb7xvpzssxyu3y775pe)
[YOLOv5 model weights needed for XtraX](https://oregonstate.box.com/s/20r8c3peu6drogsrqt3sq2cmfl5f2s3t)
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
