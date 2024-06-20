# Video Prediction

The project to combine 3 channels (video, audio, text) of CEOs' video interviews to predict the market movement.

## Divide the videos into 3 channels based on sentences 

First, please go to the subdirectory, 'Split'. 
```php
cd Split
```

Please create a new environment and activate it:
```php
conda create -n Split python=3.11 

conda activate Split
```

in Diarization.py, please use your own 'assess_token'
in main.py please use your own key of openai 

Installation of the appropriate environment
```php
pip install -r Split_requirements.txt
```

Then you can run main.py to split the original videos into 3 channels as well as images
```php
python main.py
```

Then remember to come to the root directory back!
```php
cd ..
```

## Local Features Extractor

It is used to extract the local features of video. The features can be divided into 3 groups : video, audio and text.

### text and audio
First, please go to the subdirectory, 'Local_features_extractor'. 
```php
cd Local_features_extractor
```

Please create a new environment and activate it:
```php
conda create -n Local_features_extractor python=3.11 

conda activate Local_features_extractor
```

Installation of the appropriate environment
```php
pip install -r text_audio_requirements.txt
```

Then you can run 'text_audio_features_extractor.py' to get the features of text and audio
```php
 python text_audio_extractor.py
```


### video
For video, there is another environment
```php
conda create -n video python=3.11 

conda activate video
```



The next step is also to install the requirements
```php
pip install video_requirements.txt
```

Then you can run 'video_features_extractor.py' to get the features of text and audio
```php
 python video_extractor.py
```

### eyes
For eye tracking, we need 3.7.12 python so that we create a new environment
```php
conda activate -n GazeTracking python=3.7.12
conda activate GazeTracking
```
Installation of the appropriate environment of GazeTracking

```php
pip install -r eyes_requirements.txt
```

Then you can run 'eyes_features_extractor.py'
```php
python eyes_features_extractor.py
```
Then remember to come to the root directory back!
```php
cd ..
```

## Global Features Extractor
It is used to extract the global features of video. The features can be divided into 3 groups : video, audio and text.

### video
you can still use the 'base' environment
```php
conda activate base
```


audio的分来跑
conda activate Audio
