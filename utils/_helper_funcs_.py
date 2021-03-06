from __future__ import print_function
import os,sys
from _helper_basics_ import *
from _helper_DNN_ import *
# import _models_ as mymodels

## Parsing
def get_parser(mode='inference'):

    if   mode=='dataprep':
        parser=argparse.ArgumentParser('Dataprep',
            description="Preparing the data for training.")
        parser.add_argument('-data_type')

    elif mode=='train':
        parser=argparse.ArgumentParser('Training',
            description="Training a deep hashing extractor.")
        parser.add_argument('-Archi_vrs')
        parser.add_argument('-trn_type')
        parser.add_argument('-val_type')

    elif mode=='eval':
        parser=argparse.ArgumentParser('Evaluation',
            description="Finds the performance of a test set. \
                        \nMetrics include precision, recall.\
                        \nUse different testsets for finding metrics of seen and unseen clusters."
                        )
        parser.add_argument('-Archi_vrs')
        parser.add_argument('-data_type')

    elif mode=='inference':
        parser=argparse.ArgumentParser('Hashing',
            description="Perform deep hashing on audio.")
        parser.add_argument('-Archi_vrs')
    return parser

## Dataprep
def utt2logmel(curr_utt, utt2wavpath_DICT, conf_sr=None):
    wavpath = utt2wavpath_DICT[curr_utt]
    x_wav,_=read_audio(wavpath, mode='audiofile', sr=conf_sr.sr, mean_norm=False)
    if len(x_wav) >= conf_sr.des_len:   x_wav=x_wav[:conf_sr.des_len]
    else:                               x_wav = librosa.util.fix_length(x_wav, conf_sr.des_len)
    x_LogMel=wav2Mel(x_wav, **conf_sr.kwargs_MEL)
    x_LogMel=np.expand_dims(x_LogMel, axis=-1) # numchannel = 1
    return x_LogMel

## Training
class Train_Dhash():

    def __init__(self, conf_DNN):
        self.Training_model = self.Load_Training_Model(conf_DNN)
        self.Training_model = compile_opt(
            _mod_in=self.Training_model,
            _opt_mode=conf_DNN.opt_mode,
            _opt_dict=conf_DNN.opt_dict,
            compile_dict=conf_DNN.compile_dict)
        self.CallbackLst = self.Get_Callbacks(conf_DNN)

    ## Loading Model
    def Load_Training_Model(self, conf_DNN):
        print("\n## Loading Model")
        load_model_keras_kwargs={
            'model_mode':'path',
            'model_path':conf_DNN.Training_model_path,
            'weights_path':conf_DNN.Training_weights_path,
            'verbose':True
        }
        #     'weights_path':conf_DNN.Training_weights_path,
        #     'weights_path':get_latest_file(conf_DNN.Ckpt_Mod_Weights_fold),
        self.Training_model = load_model_keras( **load_model_keras_kwargs)
        self.Training_model.summary()
        save_model( self.Training_model,
                    model_path=conf_DNN.Training_model_path,
                    weights_path=conf_DNN.Training_weights_path)
        return self.Training_model
    def Get_Callbacks(self, conf_DNN):
        reduce_lr = ReduceLROnPlateau(**conf_DNN.ReduceLROnPlateau_kwargs)
        ckpt = ckpt_saving(conf_DNN.Ckpt_Mod_Weights_fold, conf_DNN.ModelCheckpoint_kwargs, save_all=True)
        csv_log = CSVLogger(conf_DNN.csv_log_path, separator='\t', append=True)
        savemodel = SaveModel(self.Training_model, conf_DNN)
        return [reduce_lr, ckpt, csv_log, savemodel]

    ## Training Model
    def generate_samediffpairs(self, conf_DNN, cluster2utt_DICT, utt2melspect_DICT):
        batch_size=conf_DNN.batch_size
        half_size=batch_size//2
        class_list = list(cluster2utt_DICT)
        if True : # pre-initialise to reduce time
            left__inp=np.empty( (batch_size, 32, 640, 1) )
            right_inp=np.empty( (batch_size, 32, 640, 1) )
            outt_zero=np.zeros( (batch_size, 1) )
            for idx in range(batch_size):
                    outt_zero[idx]=1 if idx%2==0 else 0
        while True:
            if True : ## Get uttid pairs
                same_pair_LIST=[]
                diff_pair_LIST=[]
                curr_idx=0
                flag_break=True
                #
                for _ in range(half_size):
                    c1,c2 = random.sample( class_list , 2)
                    if True :
                        c1,c2 = random.sample( class_list , 2)
                        while len(cluster2utt_DICT[c1])<2:
                            c1,c2 = random.sample( class_list , 2)
                        #
                        same_pair = random.sample( cluster2utt_DICT[c1] , 2)
                        same_pair_LIST.append( same_pair )
                        #
                        diff_uttid=random.sample( cluster2utt_DICT[c2] , 1)
                        diff_pair_LIST.append( [same_pair[0],diff_uttid[0]] )
                #
                assert len(same_pair_LIST)==half_size, len(same_pair_LIST)
                assert len(diff_pair_LIST)==half_size, len(diff_pair_LIST)
            if True : ## Get spectrograms
                #
                idx=0
                for same_pair,diff_pair in zip(same_pair_LIST,diff_pair_LIST):
                    #
                    left__inp[idx]=utt2melspect_DICT[same_pair[0]]
                    right_inp[idx]=utt2melspect_DICT[same_pair[1]]
                    # outt_zero[idx]=1
                    idx+=1
                    #
                    left__inp[idx]=utt2melspect_DICT[diff_pair[0]]
                    right_inp[idx]=utt2melspect_DICT[diff_pair[1]]
                    # outt_zero[idx]=0
                    idx+=1
            yield (left__inp, right_inp), outt_zero

    # "WITHOUT validation data"
    def Training_woval_debug(self, conf_DNN, cluster2utt_DICT, utt2feat_DICT):
        x_tmp = next( self.generate_samediffpairs(conf_DNN, cluster2utt_DICT, utt2feat_DICT) )
        print(f"x_tmp[0][0].shape={x_tmp[0][0].shape}")
        print(f"x_tmp[0][1].shape={x_tmp[0][1].shape}")
        print(f"x_tmp[1].shape   ={x_tmp[1].shape}")
        self.Training_model.fit(
            x_tmp[0],
            x_tmp[1],
            epochs=5,
            callbacks=self.CallbackLst,
            verbose=1, )
    def Training_woval(self, conf_DNN, cluster2utt_DICT, utt2feat_DICT):
        history=self.Training_model.fit(
                        self.generate_samediffpairs(conf_DNN, cluster2utt_DICT, utt2feat_DICT),
                        steps_per_epoch=conf_DNN.steps_per_epoch,
                        epochs=conf_DNN.epochs,
                        callbacks=self.CallbackLst,
                        initial_epoch=conf_DNN.initial_epoch,
                        verbose=2,
                        )
        #                 validation_data=input_val,
        save_model( self.Training_model,
                    model_path=conf_DNN.Training_model_path,
                    weights_path=conf_DNN.Training_weights_path)
        return history

    # "WITH validation data"
    def Training_wval_debug(self, conf_DNN,
        cluster2utt_DICT, utt2feat_DICT,
        cluster2utt_DICT_val, utt2feat_DICT_val):
        x_tmp = next( self.generate_samediffpairs(conf_DNN, cluster2utt_DICT, utt2feat_DICT) )
        print(f"x_tmp[0][0].shape={x_tmp[0][0].shape}")
        print(f"x_tmp[0][1].shape={x_tmp[0][1].shape}")
        print(f"x_tmp[1].shape   ={x_tmp[1].shape}")
        x_tmp_val = next( self.generate_samediffpairs(conf_DNN, cluster2utt_DICT_val, utt2feat_DICT_val) )
        print(f"x_tmp_val[0][0].shape={x_tmp_val[0][0].shape}")
        print(f"x_tmp_val[0][1].shape={x_tmp_val[0][1].shape}")
        print(f"x_tmp_val[1].shape   ={x_tmp_val[1].shape}")

        pdb.set_trace()

        self.Training_model.fit(
            x=x_tmp[0], y=x_tmp[1],
            validation_data=(x_tmp_val[0], x_tmp_val[1]),
            epochs=5,
            callbacks=self.CallbackLst,
            verbose=1, )

        self.Training_model.fit(
            self.generate_samediffpairs(conf_DNN, cluster2utt_DICT, utt2feat_DICT),
            steps_per_epoch=5,
            epochs=5,
            validation_data=self.generate_samediffpairs(conf_DNN, cluster2utt_DICT_val, utt2feat_DICT_val),
            validation_steps=conf_DNN.validation_steps,
            callbacks=self.CallbackLst,
            verbose=1, )
    def Training_wval(self, conf_DNN,
        cluster2utt_DICT, utt2feat_DICT,
        cluster2utt_DICT_val, utt2feat_DICT_val):
        history=self.Training_model.fit(
                        self.generate_samediffpairs(conf_DNN, cluster2utt_DICT, utt2feat_DICT),
                        steps_per_epoch=conf_DNN.steps_per_epoch,
                        validation_data=self.generate_samediffpairs(conf_DNN, cluster2utt_DICT_val, utt2feat_DICT_val),
                        validation_steps=conf_DNN.validation_steps,
                        epochs=conf_DNN.epochs,
                        callbacks=self.CallbackLst,
                        initial_epoch=conf_DNN.initial_epoch,
                        verbose=2,
                        )
        save_model( self.Training_model,
                    model_path=conf_DNN.Training_model_path,
                    weights_path=conf_DNN.Training_weights_path)
        return history
class SaveModel(Callback):
    def __init__(self, Training_model, conf_DNN):
        self.Training_model = Training_model
        self.conf_DNN = conf_DNN
    def on_epoch_end(self, epoch, logs):
        print()
        if True : ## Save training model
            # print('\tSaving self.Training_model to {}\n'.format(self.conf_DNN.Training_weights_path))
            save_model(self.Training_model,
                model_path=self.conf_DNN.Training_model_path,
                weights_path=self.conf_DNN.Training_weights_path,
                verbose=False)
        if True : ## After finish training, extract submodel and save it
            # print('\tSaving model_denoise to {}\n'.format(self.conf_DNN.Enhance_weights_path))
            layer_name = self.conf_DNN.Hashing_mod_bn
            Hashing_model= Model(
                inputs=self.Training_model.get_layer(layer_name).input,
                outputs=self.Training_model.get_layer(layer_name).output)
            Hashing_model._name = layer_name
            # print('Hashing_model._name : ',Hashing_model._name)
            save_model(Hashing_model,
                model_path=self.conf_DNN.Hashing_model_path,
                weights_path=self.conf_DNN.Hashing_weights_path,
                verbose=False)

## Evaluation
class Eval_Dhash(Train_Dhash):

    def __init__(self, conf_DNN):
        super().__init__(conf_DNN)
        self.Stats_dict={'TP':0,'FN':0,'TN':0,'FP':0,'recall':'-','precision':'-','specificity':'-',}

    def update_samediff_LIST(self, lstidx,
        same_pair_LIST, diff_pair_LIST,
        cluster2utt_DICT,c1,c2):
        #
        same_pair = random.sample( cluster2utt_DICT[c1] , 2)
        same_pair_LIST[lstidx] = same_pair
        #
        diff_uttid=random.sample( cluster2utt_DICT[c2] , 1)
        diff_pair_LIST[lstidx] = [same_pair[0], diff_uttid[0]]
    def test_samediffpairs(self, conf_DNN, cluster2utt_DICT, utt2feat_DICT):
        batch_size=conf_DNN.batch_size
        half_size=batch_size//2
        class_list = list(cluster2utt_DICT)
        if True : # pre-initialise to reduce time
            left__inp=np.empty( (batch_size, 32, 640, 1) )
            right_inp=np.empty( (batch_size, 32, 640, 1) )
            outt_zero=np.zeros( (batch_size, 1) )
            for idx in range(batch_size):
                    outt_zero[idx]=1 if idx%2==0 else 0
            #
            same_pair_LIST=[None for _ in range(half_size)]
            diff_pair_LIST=[None for _ in range(half_size)]
        while True:
            if True : ## Get clusters
                for lstidx in range(half_size):
                    c1,c2 = random.sample( class_list , 2)
                    while len(cluster2utt_DICT[c1])<2: c1,c2 = random.sample( class_list , 2)
                    self.update_samediff_LIST(lstidx, same_pair_LIST, diff_pair_LIST, cluster2utt_DICT, c1, c2)
                assert not None in same_pair_LIST
                assert not None in diff_pair_LIST
            if True : ## Get spectrograms
                idx=0
                for same_pair,diff_pair in zip(same_pair_LIST,diff_pair_LIST):
                    #
                    left__inp[idx]=utt2feat_DICT[same_pair[0]]
                    right_inp[idx]=utt2feat_DICT[same_pair[1]]
                    # outt_zero[idx]=1
                    idx+=1
                    #
                    left__inp[idx]=utt2feat_DICT[diff_pair[0]]
                    right_inp[idx]=utt2feat_DICT[diff_pair[1]]
                    # outt_zero[idx]=0
                    idx+=1
            yield (left__inp, right_inp), outt_zero

    def update_TPFN(self, dict_in, curr_preds):
        # Supposed to be positive, but is labelled as negative
        if curr_preds >= 0.5:   dict_in['TP'] += 1
        else:                   dict_in['FN'] += 1
    def update_TNFP(self, dict_in, curr_preds):
        # Supposed to be negative, but is labelled as positive
        if curr_preds < 0.5:    dict_in['TN'] += 1
        else:                   dict_in['FP'] += 1
    def print_Stats(self, dict_in):
        denom1 = dict_in['TP']+dict_in['FN']
        denom2 = dict_in['TP']+dict_in['FP']
        denom3 = dict_in['TN']+dict_in['FP']
        if denom1 : dict_in['recall'] = dict_in['TP'] / denom1
        if denom2 : dict_in['precision'] = dict_in['TP'] / denom2
        if denom3 : dict_in['specificity'] = dict_in['TN'] / denom3
        print(' \tTP={}'.format(dict_in['TP']))
        print(' \tFN={}'.format(dict_in['FN']))
        print(' \tTN={}'.format(dict_in['TN']))
        print(' \tFP={}'.format(dict_in['FP']))
        print(' \t\trecall      = TP /( TP+FN )  ={}'.format(dict_in['recall']))
        print(' \t\tprecision   = TP /( TP+FP )  ={}'.format(dict_in['precision']))
        print(' \t\tspecificity = TN /( TN+FP )  ={}'.format(dict_in['specificity']))
        return

    def Evaluate(self, conf_DNN, cluster2utt_DICT, utt2feat_DICT):
        print( f"Evaluate(self, conf_DNN, cluster2utt_DICT, utt2feat_DICT)" )
        for npts, input_arr in enumerate(
            self.test_samediffpairs(conf_DNN, cluster2utt_DICT, utt2feat_DICT)
            ):
            if npts==100: break
            pred_all = self.Training_model.predict(input_arr[0], batch_size=conf_DNN.batch_size)
            for curr_label, curr_preds in zip(input_arr[1], pred_all):
                if  curr_label : # curr_label == 1: # P , same
                    self.update_TPFN(self.Stats_dict, curr_preds)
                else           : # curr_label == 0: # N , diff
                    self.update_TNFP(self.Stats_dict, curr_preds)
        self.print_Stats(self.Stats_dict)
        print( f"Done Evaluation" )
        return

## Inference
class Dhash():

    def __init__(self, conf_DNN):
        self.Hashing_model = self.Load_model(conf_DNN)

    def Load_model(self, conf_DNN):
        print("\n## Loading Model")
        load_model_keras_kwargs={
            'model_mode':'path',
            'model_path':conf_DNN.Hashing_model_path,
            'weights_path':conf_DNN.Hashing_weights_path,
            'verbose':True
        }
        self.Hashing_model=load_model_keras(**load_model_keras_kwargs)
        self.Hashing_model.summary()
        return self.Hashing_model

    def Hashing(self, x_wav, conf_sr):
        # Trim the audio length
        if len(x_wav) >= conf_sr.des_len:   x_wav=x_wav[:conf_sr.des_len]
        else:                               x_wav=librosa.util.fix_length(x_wav, conf_sr.des_len)

        # Compute Log Mel feature
        x_LogMel=wav2Mel(x_wav, **conf_sr.kwargs_MEL)
        x_LogMel=np.expand_dims(x_LogMel, axis=0) # batch_size = 1
        x_LogMel=np.expand_dims(x_LogMel, axis=3) # numchannel = 1

        # Perform Hashing
        return self.Hashing_model.predict(x_LogMel)[0]

"""
!import code; code.interact(local=vars())
"""
