#!/usr/bin/bash
conda activate v2_Py36_tfgpu2
<< COMMENTS
    python                  3.6
    tensorflow-gpu          2.0.0
    numpy                   1.18.1
    librosa                 0.7.2
        if librosa <= 0.7.1 : use norm=1 instead of norm='slaney'
COMMENTS

Archi_vrs=SSResCNN
trn_type=train

echo "Archi_vrs = $Archi_vrs"
echo "trn_typ   = $trn_type"
