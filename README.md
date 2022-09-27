# End-to-end Safety Detection Pipeline
## Installation
### Download models
Download following pre-trained models to the specified paths:
https://drive.google.com/file/d/1Yb3LTVYQSWeS3EAZ2XrEw0xMewua2Qz9/view?usp=sharing 
https://drive.google.com/file/d/18RCemKMiJzsO2sfUwWiBB-LHI4Jbq8Oh/view?usp=sharing
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
