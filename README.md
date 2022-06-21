# End-to-end Safety Detection Pipeline

## Downloading
Download this repo and add following files to the specified paths:
https://drive.google.com/file/d/1wUViXZ-qEPQBvv_SR1AnYHGcHgkjm_fb/view?usp=sharing to the root path.


<b> Check constants.py file for more details </b>

## Reference Area
By using reference_area.py, get reference area coordinates and paste it to safety-detection/constants.py
You can run this script with the following command:
```
python reference_area.py --video_path <video path>
```
You will start selecting from left bottom point and continue counter-clockwise.

## Running End-to-End Pipeline
After changing paths in the safety-detection/constants.py, run Demo.ipynb notebook.
* SOURCE variable is for video path.
* REFERENCE_AREA variable is for reference area coordinates.