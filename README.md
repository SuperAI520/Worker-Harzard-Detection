# End-to-end Safety Detection Pipeline
## Installation
### Download models
Download following pre-trained models to the specified paths:

https://drive.google.com/file/d/1PWB60MR3FrZBj0UudmeDOne7bo34UVjF/view?usp=drive_link

https://drive.google.com/file/d/1gZSm-Te9MW3CLMML3bt7BMu1YdeKSpQa/view?usp=drive_link

https://drive.google.com/file/d/17UfQ0o1uzkb3LgdqI5cm3QjPgFjVlAQc/view?usp=drive_link

to the /safety-detection/models

To get permission for AI models, contact zlingxiao1028@gmail.com or Whatsapp: +14158003112

<b> Check constants.py file for more details </b>

### Install
```
# Clone Safety Detection repository
git clone https://github.com/JasonJin211/Safety-Detection.git

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

https://github.com/JasonJin211/Safety-Detection/assets/60502049/8f7bc5a5-e2a7-493f-add7-a04f47e96e84

https://github.com/JasonJin211/Safety-Detection/assets/60502049/56ac6d2d-ac37-4f07-93bc-dd0eefdf2466






