# Depth and motion learning

Forked from:
- https://github.com/google-research/google-research/tree/master/depth_and_motion_learning
- https://github.com/PhilippSchmaelzle/mono_depth/tree/inference_with_original_code_structure

## Struttura cartella

In depth_and_motion_learning sono presenti tutti i file python che vengono chiamati dai 2 script bash presenti nella directory principale.

## Installazione 

- Scaricare repo da: 
https://github.com/google-research/google-research/tree/master/depth_and_motion_learning
- Vedere file [tensorflow-from-source-guide.md](https://github.com/carloradice/tesi/blob/main/tensorflow-from-source-guide.md)
- CUDA 10.1.243 
- CUDNN versione 7.6.5
- In `tensorflow/config.py` quando viene fatto il build di tensorflow aggiungere questo codice alla riga 1357

`  config = {
'cublas_include_dir': '/home/radice/customCuda/cuda-10.1.243/include',
'cublas_library_dir': '/usr/lib/x86_64-linux-gnu',
'cuda_binary_dir': '/home/radice/customCuda/cuda-10.1.243/bin',
'cuda_include_dir': '/home/radice/customCuda/cuda-10.1.243/include',
'cuda_library_dir': '/home/radice/customCuda/cuda-10.1.243/lib64',
'cuda_toolkit_path': '/home/radice/customCuda/cuda-10.1.243/', 
'cuda_version': '10.1', 
'cudnn_include_dir': '/home/radice/customCuda/cuda-10.1.243/include',
'cudnn_library_dir': '/home/radice/customCuda/cuda-10.1.243/lib64',
'cudnn_version': '7.6.5',
'cupti_include_dir': '/home/radice/customCuda/cuda-10.1.243/extras/CUPTI/include',
'cupti_library_dir': '/home/radice/customCuda/cuda-10.1.243/extras/CUPTI/lib64',
'nvvm_library_dir': '/home/radice/customCuda/cuda-10.1.243/nvvm/libdevice'
}
`

### Requirements

- python 3.6 
- pip conda (https://anaconda.org/anaconda/pip)
- opencv2

Tramite pip conda:

- bazel conda 0.26.1 (https://anaconda.org/conda-forge/bazel)
- numpy 18.5 (1.19.5 dopo aver installato opencv2)
- matplotlib==3.3.0            
- tensorflow-graphics==1.0.0   

### Scaricare checkpoint Imagenet 

Seguendo la guida di :
https://github.com/google-research/google-research/issues/589

#### Requirements
Tramite conda:

- python < 3.7
- tensorflow==1.8.0 (conda install -c conda-forge tensorflow==1.8)
- torchfile (conda install -c conda-forge torchfile)

#### Modifiche
- cambiare: `import cPickle as pickle` --> `import pickle as pickle`

## Dataset

### Oxford
Scaricare dataset da: https://robotcar-dataset.robots.ox.ac.uk/

Effettuare preprocessing.

### Kitti 

Struttura del dataset
```
.
├── data
│   ├── 2011_09_26
│   │   ├── 2011_09_26_drive_*
│   │   │   ├── image_02
│   │   │   │   ├── data
│   │   │   │   │   └── *.png
│   │   │   │   └── timestamps.txt
│   │   │   └── image_03
│   │   │       ├── data
│   │   │       └── timestamps.txt
│   │   │  
│   │   ├── calib_cam_to_cam.txt
│   │   ├── calib_imu_to_velo.txt
│   │   └── calib_velo_to_cam.txt
│   ├── 2011_09_28
│   ├── 2011_09_29
│   ├── 2011_09_30
│   └── 2011_10_03
├── mask-rcnn
│   ├── 2011_09_26_drive_*
│   │   ├── image_02
│   │   │   └── *.npz
│   │   └── image_03
│   ├── 2011_09_28
│   ├── 2011_09_29
│   ├── 2011_09_30
│   └── 2011_10_03
└── struct2depth
          ├── 2011_09_26_drive_*
          │   ├── image_02
          │   │   └── *.png
          │   │   └── *-fseg.png
          │   │   └── *-cam.txt
          │   └── image_03
          ├── 2011_09_28
          ├── 2011_09_29
          ├── 2011_09_30
          └── 2011_10_03
```

## Costruzione dataset di training
 
Guardare [struct2depth](https://github.com/tensorflow/models/tree/archive/research/struct2depth) 
per il formato delle immagini di training.

Nota: prima di esegurire lo script generare le maschere delle immagini tramite 
[mask-rcnn](https://github.com/carloradice/mask-rcnn). 

Generare il dataset, ad esempio per Oxford Robot Car, con il comando:

```shell
python depth_and_motion_learning/generator/gen_data_oxford.py --folder
```

Per generare il file **train.txt** contenente i percorsi alle immagini di 
training, eseguire:

```shell
python depth_and_motion_learning/generator/splits_generator.py --folder --dataset
```

Per generare file di train composti da più routes, dopo aver generato i relativi file **train.txt**, 
eseguire:

```shell
python depth_and_motion_learning/generator/splits_mixer.py --list --dataset
```

## Training

```shell
bash run_train.sh
```

## Testing

```shell
bash run_predict.sh
```
