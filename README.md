# Listen, Attend and Spell - PyTorch Implementation
               
My first project of Speech recognition. 
This is a PyTorch implementation of[ Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2) (LAS) and based on [Alexander-H-Liu](https://github.com/Alexander-H-Liu/Listen-Attend-and-Spell-Pytorch)' repository .



## Requirements
* Python 3
* PyTorch 1.0.0
* [python\_speech\_features](https://github.com/jameslyons/python_speech_features)
* editdistance

## Chinese Mandarin corpus
* [THCHS-30](http://www.openslr.org/18/)

* [Aishell](http://www.openslr.org/33/)
* [Primewords Chinese Corpus Set 1](http://www.openslr.org/47/)
* [Free ST Chinese Mandarin Corpus](http://www.openslr.org/38/)

## Pretrained models (not supported)


## Setup

### Download four datasets and preprocessing

```
├── audio_data
│   ├── data_thchs30
│   │   ├── data
│   │   ├── train
│   │   │   ├── ...
│   ├── data_aishell
│   │   ├── transcript
│   │   ├── wav
│   │   │   ├── ...
│   ├── primewords_md_2018_set1
│   │   ├── audio_files
│   │   ├── set1_transcript.json
│   ├── ST-CMDS-20170001_1-OS
│   │   │   ├── ...
│   ├── ...
```
 we should invoke the ```util/dict_zh_words.py``` script first, generating Chinese Dict. we can now invoke the ```util/preprocess_all_datasets.py``` script, which will read all of this in and create four pickle files. Then, invoke the ```util/load_datasets.py``` script.
 
```python
 $ python util/dict_zh_words.py
 $ python util/preprocess_all_datasets.py 
 $ python util/load_datasets.py 
```
### Start training
```python
bash train.sh
```

### Evaluate on test split

## Acknowledgements
Thanks the original [LAS](https://arxiv.org/abs/1508.01211v2), [Alexander-H-Liu](https://github.com/Alexander-H-Liu/Listen-Attend-and-Spell-Pytorch) and awesome PyTorch team.