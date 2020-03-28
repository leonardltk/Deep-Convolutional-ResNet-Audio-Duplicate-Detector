# Deep-Convolutional-ResNets-Audio-Duplicate-Detector

## Init
```bash
. path.sh
```

## Dataprep
```bash
python ./utils/step1_dataprep.py -data_type $trn_type
```

## Model Setup
```bash
python ./utils/step2_modelsetup.py -Archi_vrs $Archi_vrs
```

## Train
```bash
python train.py -Archi_vrs $Archi_vrs -trn_type $trn_type
```

## Inference
```bash
python inference.py -Archi_vrs $Archi_vrs
```
