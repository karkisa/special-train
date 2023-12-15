# Special-Train
## Deployment using lightning for getting frames and segmentation.

### Cone repository

```
git clone https://github.com/karkisa/special-train.git
```
### Create Environment
Let's create an environment "Whael_Morph" (You are free to use any other name for your environment)

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
[Yolov5 model weights needed for DeteX](https://oregonstate.box.com/s/4bl2pr0xuygbai8gu97hajjs0ihprc7w)


[SAM model weights neede for XtraX](https://oregonstate.box.com/s/oltsl30mxvmqvsb7xvpzssxyu3y775pe)


[YOLOv5 model weights needed for XtraX](https://oregonstate.box.com/s/20r8c3peu6drogsrqt3sq2cmfl5f2s3t)


## Quick tutorial

Follow this video [tutorial](https://kaltura.oregonstate.edu/media/1_ssnzylci) for DeteX and XtraX application use

### launch DeteX

```
lightning run app DeteX.py
```

The files saved by DeteX follows the format: 
VideoName_000_HH_MM_SS_000_HM_HC.png 

"_000_" is a saperator that helps the appliation identify different information from the file name.

“VideoName” is the name of the video from which it was selected. HH_MM_SS is the time stamp where HH refers to hours, MM refers to minutes and SS refers to seconds. HM_HC captures altitude information. For example, if the height is. 45.6 then HM is 45 and HC is 6.  

The applcation lets you select folders which you want to analyse and folder where you want to save the results. For each video the application creates new folder named as per the video name where it saves frames from that video.

### launch XtraX
To run XtraX on just one image and have more control use "app_sam.py" as below
```
lightning run app app_sam.py
```

To perform extraction on folder containing selected frames use "XtraX.py"

```
lightning run app XtraX.py
```

The results folder contains a csv "results.csv"
The folloing is the discription of the colums that the csv contains

'image'                : image path
'model_length_meters'  : total length of whale in meters  
'model_length_pixel'   : totla length in pixels
'pixels_roi'           : Surface area of region of interest in pixel squared
'bai'                  : BAI calculated using the prediction
'centers'              : Python dictionary that contains the (x,y) pixel coordinates with keys representing the points on the whale
'polygon_full'         : Polygon represent the binary mask of whole whale in COCO format 
'total_whale_area'     : Total area of whale in pixels          
'sensor_width'         : Sensor width of the camera used
'launch_height'        : Launch height of the drone
'focal_length'         : Focal length of the camera used       
'altitude'             : Altitude of the drone in meters. (This is extracted from the name of the file. It also adds launch height to the value from name. If the name does not have altitude information then it defaults to 50 m)
'model_50_length_pixel': 50% of total length of whale in pixels.