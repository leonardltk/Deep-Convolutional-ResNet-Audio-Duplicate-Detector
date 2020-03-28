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
    parser=get_parser(mode='dataprep')
    args=parser.parse_args()
    conf_sr=conf.SR_conf()
    conf_data=conf.Data_conf(args.data_type)
    str(conf_sr)
    str(conf_data)

if True: ## Input texts
    cluster2utt_DICT=dict( 
        (i.split(' ')[0], i.split(' ')[1:]) for i in 
            [line.strip('\n') for line in open(conf_data.cluster2utt,'r')]
    )
    utt2cluster_DICT=dict( line.strip('\n').split(' ') for line in open(conf_data.utt2cluster,'r') )
    utt2wavpath_DICT=dict( line.strip('\n').split(' ') for line in open(conf_data.wav_scp,'r') )
    dump_load_pickle(conf_data.cluster2utt_DICT_path, 'dump', cluster2utt_DICT)
    dump_load_pickle(conf_data.utt2cluster_DICT_path, 'dump', utt2cluster_DICT)
    dump_load_pickle(conf_data.utt2wavpath_DICT_path, 'dump', utt2wavpath_DICT)

if True: ## Output feats
    utt2feat_DICT=dict( 
        (utt, utt2logmel(utt, utt2wavpath_DICT, conf_sr)) 
        for utt,cluster in utt2cluster_DICT.items()
        )
    dump_load_pickle(conf_data.utt2melspect_DICT_path, 'dump', utt2feat_DICT)


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
