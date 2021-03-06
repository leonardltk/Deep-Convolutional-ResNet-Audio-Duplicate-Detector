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
val_type=val
test_type_seen=test_seen
test_type_unseen=test_unseen

echo "Archi_vrs         = $Archi_vrs"
echo "trn_type          = $trn_type"
echo "val_type          = $val_type"
echo "test_type_seen    = $test_type_seen"
echo "test_type_unseen  = $test_type_unseen"
