from __future__ import print_function
if True: ## imports / admin
    import os,sys
    sys.path.insert(0, 'utils')
    from _helper_funcs_ import *
    import conf
    # 
    START_TIME=datetime.datetime.now()
    print(f"===========\npython {' '.join(sys.argv)}\n Start_Time:f{START_TIME}\n===========")

    print('############ Config Params ############')
    parser=get_parser(mode='train')
    args=parser.parse_args()
    conf_DNN=conf.DNN_conf(args.Archi_vrs)
    conf_sr=conf.SR_conf()
    conf_trn=conf.Data_conf(args.trn_type)

if True : ######################## Load Data ###############################
    cluster2utt_DICT=dump_load_pickle(conf_trn.cluster2utt_DICT_path, 'load')
    utt2cluster_DICT=dump_load_pickle(conf_trn.utt2cluster_DICT_path, 'load')
    utt2wavpath_DICT=dump_load_pickle(conf_trn.utt2wavpath_DICT_path, 'load')
    utt2feat_DICT=dump_load_pickle(conf_trn.utt2melspect_DICT_path, 'load')

if True : ######################## Training ###############################
    train_d = Train_Dhash(conf_DNN)
    try:
        x_dhash=train_d.Training(conf_DNN, cluster2utt_DICT, utt2feat_DICT)
    except:
        print('\n!!!\n')
        traceback.print_exc()
        save_model(Training_model,
                model_path=conf_DNN.Training_model_path+'.error',
                weights_path=conf_DNN.Training_weights_path+'.error',
                verbose=True)

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
