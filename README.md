# Special-Train
## Deployment using lightning for getting frames and segmentation.


### Clone repository


```
git clone https://github.com/karkisa/special-train.git
```
### Create Environment
Let's create an environment "Whale_Morph" (You are free to use any other name for your environment)


```
conda create --name Whale_Morph python=3.8


```


make the repo folder the base folder in terminal


```
cd special-train
```


Use the environment.yaml file to get all the packages needed for the application
These applications may have new features and therefore may have additional dependencies. To run the application smoothly update the environment in which you will be running this application using the following command


```
conda env update --name Whale_Morph --file environment.yml


```


### Activate Environment
```
conda activate Whale_Morph
```


### download data


Before you run these applications you need to download below mentioned files and place them in the repository folder.


[Yolov5 model weights needed for DeteX](https://oregonstate.box.com/s/4bl2pr0xuygbai8gu97hajjs0ihprc7w)


[YOLOv5 model weights needed for keypoint identification in XtraX](https://oregonstate.box.com/s/20r8c3peu6drogsrqt3sq2cmfl5f2s3t)


[YOLOv8 segmentation model needed for XtraX](https://oregonstate.box.com/s/fedsup6yhfoi7epx7gucgfexl71amqp7)


[SAM model weights needed for XtraX](https://oregonstate.box.com/s/oltsl30mxvmqvsb7xvpzssxyu3y775pe)


## Quick tutorial


Follow this video [tutorial](https://media.oregonstate.edu/media/1_01o1wp56) for DeteX and XtraX application use


### launch DeteX


```
lightning run app DeteX.py
```


The files saved by DeteX follows the format:
VideoName_000_HH_MM_SS_000_HM_HC.png


"_000_" is a separator that helps the application identify different information from the file name.


“VideoName” is the name of the video from which it was selected. HH_MM_SS is the time stamp where HH refers to hours, MM refers to minutes and SS refers to seconds. HM_HC captures altitude information. For example, if the height is. 45.6 then HM is 45 and HC is 6. 


The application lets you select the folders which you want to analyze and the folder where you want to save the results. For each video, the application creates a new folder named as per the video name where it saves frames from that video.


For more user control and analysing a single video use the following command
```
lightning run app app_detection_finetune.py
```
In this application, you need to manually add the path to the video and folder.


### launch XtraX




* To perform extraction on a folder containing selected frames use "XtraX.py".
* Note: XtraX is designed to extract altitude information from the file's name. It expects the file name to be in the same format as the DeteX saves it to be able to extract the altitude information. If the file name does not contain altitude information, the default altitude is set to 50 m. Users can then use the output length and area measurements in pixels in the results.csv to manually convert to meters using the correct altitude since DeteX automatically embeds the barometer altitude within the file name. This may be particularly helpful to recalculate measurements using altitude recorded with a LiDAR altimeter.


```
lightning run app XtraX.py
```

The results folder contains a CSV "results.csv"
The following is the description of the columns that the CSV contains

* 'image_file_name'      : image file name
* 'image_file_path       : image fiel path
* 'sensor_width'         : Sensor width of the camera used
* 'launch_height'        : Launch height of the drone
* 'focal_length'         : Focal length of the camera used      
* 'altitude'             : The altitude of the drone in meters. (This is extracted from the name of the file. It also adds launch height to the value from the name. If the name does not have altitude information then it defaults to 50 m)
* 'Head-Tail-start'      : Starting width percentage of the Head-Tail range
* 'Head-Tail-end'        : Ending width percentage of the Head-Tail range
* 'model_HT_length_pixel': Length between Head-Tail range of whale in pixels
* 'model_length_meters'  : total length of whale in meters 
* 'model_length_pixel'   : total length in pixels
* 'pixels_roi'           : Surface area of the region of interest in pixel squared
* 'bai'                  : BAI calculated using the predictions
* 'total_whale_area'     : Total area of whale in pixels         
* 'centers'              : Python dictionary that contains the (x,y) pixel coordinates with keys representing the points on the whale
* 'polygon_full'         : Polygon represents the binary mask of the whole whale in COCO format



To run XtraX on just one image and have more control use "app_sam.py" as below
```
lightning run app app_sam.py
```


##How to close the applications
If you are using a Mac, then just use "control+C" in the terminal for Windows "windows+C" in the terminal


Feel free to reach out to me at karkisa@oregonstate.edu


