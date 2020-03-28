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
    parser=get_parser('API')
    args=parser.parse_args()
    conf_DNN=conf.DNN_conf(args.Archi_vrs)
    conf_sr=conf.SR_conf()


if 1 : ######################## Load Audio ###############################
    wavpath="./wav/test.mp3"
    x_wav,_=read_audio(wavpath, mode='audiofile', sr=conf_sr.sr, mean_norm=False)

if 1 : ######################## Perform Hashing ###############################
    d = Dhash(conf_DNN) # init
    x_dhash=d.Hashing(x_wav, conf_sr)
    print(x_dhash.shape)


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
