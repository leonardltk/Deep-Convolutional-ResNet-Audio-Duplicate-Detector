from __future__ import print_function
if 1: ## imports / admin
    import os,sys,pdb
    sys.path.insert(0, 'utils')
    from _helper_funcs_ import *
    import _models_ as mymodels
    import conf
    # 
    START_TIME=datetime.datetime.now()
    datetime.datetime.now() - START_TIME
    print(f"===========\npython {' '.join(sys.argv)}\n Start_Time:f{START_TIME}\n===========")

    print('############ Config Params ############')
    parser=get_parser()
    args=parser.parse_args()
    conf_DNN=conf.DNN_conf(args.Archi_vrs)
    conf_sr=conf.SR_conf()

if 1 : ## Create Hashing Network
    Hashing_model=mymodels.f_model(conf_DNN.input_shape)
    Hashing_model._name=conf_DNN.Hashing_mod_bn
    Hashing_model.summary()
    save_model(Hashing_model, 
        model_path=conf_DNN.Hashing_model_path, 
        weights_path=conf_DNN.Hashing_weights_path)

if 1 : ## Create siamese Network
    Training_model=mymodels.f_siamese(Hashing_model, conf_DNN.input_shape)
    Training_model._name=conf_DNN.Training_mod_bn
    Training_model.summary()
    save_model( Training_model, 
        model_path=conf_DNN.Training_model_path, 
        weights_path=conf_DNN.Training_weights_path)


#################################################################
END_TIME=datetime.datetime.now()
print(f"===========\
    \nDone \
    \npython {' '.join(sys.argv)}\
    \nStart_Time  :{START_TIME}\
    \nEnd_Time    :{END_TIME}\
    \nDuration    :{END_TIME-START_TIME}\
\n===========")

"""
!import code; code.interact(local=vars())
"""
