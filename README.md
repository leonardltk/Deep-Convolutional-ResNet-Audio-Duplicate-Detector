# Deep-Convolutional-ResNets-Audio-Duplicate-Detector

## Init
```bash
# This initialises all the required environment and global variables for the following scripts.
. path.sh
```

## Dataprep
```bash
python ./utils/step1_dataprep.py -data_type $trn_type &
python ./utils/step1_dataprep.py -data_type $val_type &
python ./utils/step1_dataprep.py -data_type $test_type_seen &
python ./utils/step1_dataprep.py -data_type $test_type_unseen &
```

## Model Setup
```bash
python ./utils/step2_modelsetup.py -Archi_vrs $Archi_vrs
```

## Train
```bash
python train.py -Archi_vrs $Archi_vrs -trn_type $trn_type -val_type $val_type
```

## Evaluation
```bash
python eval.py -Archi_vrs $Archi_vrs -data_type $test_type_seen
python eval.py -Archi_vrs $Archi_vrs -data_type $test_type_unseen
```

## Inference
```bash
python inference.py -Archi_vrs $Archi_vrs
```
