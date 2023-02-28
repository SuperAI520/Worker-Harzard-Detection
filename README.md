# End-to-end Safety Detection Pipeline
## Installation
### Download models
Download following pre-trained models to the specified paths:
https://drive.google.com/file/d/1iizEQjiTetEzaBuzLqDgrWjFZL9XXw0n/view?usp=share_link
https://drive.google.com/file/d/1hLolnZNNe_2rU2NitwYHt1Yw5bM2d68K/view?usp=share_link

https://drive.google.com/file/d/1g646DkMzZ1mRk0CJIJxZfcnzWdaX8DU3/view?usp=share_link

https://drive.google.com/file/d/1A70i9-OdeN84TO2p_sXZ8eR8Mb3vXiGl/view?usp=share_link

https://drive.google.com/file/d/1M9b_Zt3zL-Gx8Yl7ZbHm8lro_hfL3sN1/view?usp=share_link
to the /safety-detection/models

<b> Check constants.py file for more details </b>

### Install
```
# Clone Safety Detection repository
git clone https://github.com/groundup-ai/optimization_cv_suspended_load.git

# Install other dependencies
cd safety-detection
python -m pip install -r requirements.txt

# Install mmsegmentation dependencies
cd mmsegmentation
python -m pip install -v -e .
```

## Use built Docker images
If you do not have a Docker environment, please refer to https://www.docker.com/.

Once you installed Docker environment, you can build and run a docker container named 'alex:latest' with the following commands:

```
docker build . -t alex:latest

docker run --gpus all -it -v $(pwd)/safety-detection:/workspace/safety-detection --rm alex:latest

# Install mmsegmentation dependencies
cd mmsegmentation
python -m pip install -v -e .
```

## Inference
After completing the environment installation, you can run the Safety Detection Pipeline with the following command:

### Hatch environment
```
cd safety-detection
taskset --cpu-list 0 python clip_object_tracker.py --source videos/sample.avi --ignored_classes chain Forklift --danger_zone_width_threshold 800 --danger_zone_height_threshold 200 --int8
```

### Wharf environment
```
cd safety-detection
taskset --cpu-list 0 python clip_object_tracker.py --source videos/sample.avi --ignored_classes chain Forklift --danger_zone_width_threshold 800 --danger_zone_height_threshold 200 --wharf --int8
```

An output video will be generated under the ```run/detect``` folder. The result is as shown below:


https://user-images.githubusercontent.com/95361430/192615813-c3c89675-3e87-437b-bf18-12364063ba7a.mp4


