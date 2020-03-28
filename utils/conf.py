from __future__ import print_function
import os


class Data_conf():
    def __init__(self, datatype):
        self.datatype=datatype
        ############ Data Dir/Path ############
        self.data_dir=os.path.join('./data',self.datatype); os.makedirs( self.data_dir ,exist_ok=True)
        self.wav_scp=os.path.join(self.data_dir,'wav.scp')
        self.cluster2utt=os.path.join(self.data_dir,'cluster2utt')
        self.utt2cluster=os.path.join(self.data_dir,'utt2cluster')
        ############ Exp Dir/Path ############
        self.exp_dir=os.path.join('./exp',self.datatype)
        self.exp_dict_dir=os.path.join(self.exp_dir, 'dict'); os.makedirs( self.exp_dict_dir ,exist_ok=True)
        if 1: ## dict
            self.cluster2utt_DICT_path=os.path.join(self.exp_dict_dir,'cluster2utt.dict')
            self.utt2cluster_DICT_path=os.path.join(self.exp_dict_dir,'utt2cluster.dict')
            self.utt2wavpath_DICT_path=os.path.join(self.exp_dict_dir,'utt2wavpath.dict')
            self.utt2melspect_DICT_path=os.path.join(self.exp_dict_dir,'utt2melspect.dict')
        ############ Wav Dir/Path ############
        self.wav_dir=os.path.join('./wav', self.datatype); os.makedirs( self.wav_dir ,exist_ok=True)
        ############ Results Dir/Path ############
        self.results_dir=os.path.join('./results', self.datatype); os.makedirs( self.results_dir ,exist_ok=True)

    def __str__(self):
        print(f'\n############ Data_conf ############')
        print(f'\t\t datatype               : {self.datatype}')
        print(f'\t############ Data Dir/Path ############')
        print(f'\t\t data_dir               : {self.data_dir}')
        print(f'\t\t wav_scp                : {self.wav_scp}')
        print(f'\t\t cluster2utt            : {self.cluster2utt}')
        print(f'\t\t utt2cluster            : {self.utt2cluster}')
        print(f'\t############ Exp Dir/Path ############')
        print(f'\t\t exp_dir                    : {self.exp_dir}')
        print(f'\t\t   exp_dict_dir             : {self.exp_dict_dir}')
        print(f'\t\t     cluster2utt_DICT_path  : {self.cluster2utt_DICT_path}')
        print(f'\t\t     utt2cluster_DICT_path  : {self.utt2cluster_DICT_path}')
        print(f'\t\t     utt2wavpath_DICT_path  : {self.utt2wavpath_DICT_path}')
        print(f'\t############ Wav Dir/Path ############')
        print(f'\t\t wav_dir                : {self.wav_dir}')
        print(f'\t############ Results Dir/Path ############')
        print(f'\t\t results_dir            : {self.results_dir}')
        return ""


class SR_conf():
    def __init__(self):
        ############ Audio Wave Parameters ############
        self.sr=8000
        self.n_fft=256
        self.win_length=256
        self.hop_length=self.n_fft//2
        self.winlen=1.*self.win_length/self.sr
        self.winstep=1.*self.hop_length/self.sr
        self.num_freq_bins=self.n_fft//2 +1

        self.n_mels=32

        # self.desired_power=6.392e-05
        self.des_len = 81536
        ############ Audio Feat Parameters ############
        self.kwargs_MEL={
            'mode':"librosa",
            'sr_in':self.sr,
            'pad_mode':True,
            'n_fft':self.n_fft,
            'hop_length':self.hop_length,
            'n_mels':self.n_mels,
        }

    def __str__(self):
        print('\n############ SR_conf ############')
        print('\t############ Audio Wave Parameters ############')
        print(f'\t\t sr                 :{self.sr}')
        print(f'\t\t n_fft              :{self.n_fft}')
        print(f'\t\t win_length         :{self.win_length}')
        print(f'\t\t hop_length         :{self.hop_length}')
        print(f'\t\t winlen             :{self.winlen}')
        print(f'\t\t winstep            :{self.winstep}')
        print(f'\t\t num_freq_bins      :{self.num_freq_bins}')

        print(f'\t\t n_mels             :{self.n_mels}')

        # print(f'\t\t desired_power      :{self.desired_power}')
        print(f'\t\t des_len            :{self.des_len}')
        print('\t############ Audio Feat Parameters ############')
        print(f'\t\t kwargs_MEL         :{self.kwargs_MEL}')
        return ""


class DNN_conf():
    def __init__(self, Archi_vrs="SSResCNN", ):
        self.Archi_vrs=Archi_vrs
        ############ Dir ############
        if True:
            self.Archi_dir = os.path.join('model', self.Archi_vrs)
            self.Weights_path            = os.path.join(self.Archi_dir, "Logs");                       os.makedirs(self.Weights_path,exist_ok=True)
            self.Ckpt_Mod_Weights_fold   = os.path.join(self.Archi_dir, "Checkpoint_Model_Weights");   os.makedirs(self.Ckpt_Mod_Weights_fold,exist_ok=True)
            self.Final_Weights_fold      = os.path.join(self.Archi_dir, "Final_Model_Weights");        os.makedirs(self.Final_Weights_fold,exist_ok=True)
            if True:
                # Hashing Model
                self.Hashing_mod_bn=self.Archi_vrs+'_Hash'
                self.Hashing_model_path   = os.path.join(self.Final_Weights_fold,self.Hashing_mod_bn+'.json')
                self.Hashing_weights_path = os.path.join(self.Final_Weights_fold,self.Hashing_mod_bn+'.h5')
                # Training Model
                self.Training_mod_bn=self.Archi_vrs+'_Train'
                self.Training_model_path   = os.path.join(self.Final_Weights_fold,self.Training_mod_bn+'.json')
                self.Training_weights_path = os.path.join(self.Final_Weights_fold,self.Training_mod_bn+'.h5')
            self.Plot_path_dir           = os.path.join(self.Archi_dir,'Plots');                      os.makedirs(self.Plot_path_dir,exist_ok=True)
            self.Plot_path = os.path.join(self.Plot_path_dir,'{}.png'.format(Archi_vrs))
        ############ Model Parameters ############
        if True: ## Model input/output shapes
            self.input_shape=(32,640,1)
            self.output_shape=1
            self.mdl_name=f"{self.Archi_vrs}"
        if True: ## Regularization
            self.reg_mode="l1_l2"
            self.reg_l1=1e-4
            self.reg_l2=1e-4
            self.kwargs_reg={}
            self.kwargs_reg['reg_l1']=self.reg_l1
            self.kwargs_reg['reg_l2']=self.reg_l2
        ######################## Hyper Parameters ###############################
        if True:
            self.batch_size = 32;
            self.initial_epoch = 1-1;

            self.epochs = 300;
            self.steps_per_epoch = 10000;

            self.validation_steps = 100;
        ######################## Optimizer, Metrics Parameters ###############################
        if True: ## Optimizer
            self.opt_mode='Adam'
            self.opt_dict={'lr':1e-3}
            if self.opt_mode=='SGD':
                self.opt_dict['momentum']=.99
                self.opt_dict['decay']=0
            elif self.opt_mode=='Adam':
                self.opt_dict['decay']=0
                self.opt_dict['beta_1']=0.9
                self.opt_dict['beta_2']=0.999
                self.opt_dict['amsgrad']=False
        if True: ## Loss, Metrics,
            self.compile_dict = {}
            #
            self.loss_type='huber_loss' # 'mse' 'mae'
            self.compile_dict['loss'] = self.loss_type
            #
            self.metrics=None # 'categorical_accuracy' 'acc'
            self.compile_dict['metrics'] = self.metrics
            #
            self.compile_dict['loss_weights']=[1]
            #
            # self.compile_dict['sample_weight_mode']=None
            # self.compile_dict['weighted_metrics']=None
            # self.compile_dict['target_tensors']=None
        ######################## Callbacks ###############################
        if True: 
            self.Callbacks_list=['ReduceLROnPlateau','ModelCheckpoint','CSVLogger']
            
            ## ReduceLROnPlateau
            self.ReduceLROnPlateau_kwargs={
                "monitor":'loss',
                "factor":0.5,
                "patience":5,
                "verbose":1,
                "mode":'auto',
                "min_delta":1e-6,
                "cooldown":3,
                "min_lr":1e-6,
                }
            
            ## ModelCheckpoint
            self.CkptFold_det = [self.Archi_vrs, self.Ckpt_Mod_Weights_fold]
            self.ModelCheckpoint_kwargs={
                "monitor" : 'loss', 
                "verbose" : 1, 
                "save_best_only":True, 
                "save_weights_only":False,
                "mode":'auto',
                }

            ## CSVLogger
            self.csv_log_path = os.path.join(self.Weights_path,'{}_Trglog.txt'.format(self.Archi_vrs))
            
            ## Save_Live_Plot
        ######################## Inference Details ###############################
        if True: 
            self.expinf_dir = os.path.join('exp_inf')
            self.expinf_wav_dir = os.path.join(self.expinf_dir, self.Archi_vrs); os.makedirs(self.expinf_wav_dir,exist_ok=True)
        ######################## Results Details ###############################
        if True: 
            self.results_dir = os.path.join('results')
            self.results_vrs_dir = os.path.join(self.results_dir,self.Archi_vrs); os.makedirs(self.results_vrs_dir,exist_ok=True)

    def __str__(self):
        print(f'\n############ {self.Archi_vrs}_conf ############')
        print('\t############ Dir ############')
        if True:
            print(f'\t\t Archi_dir                  :{self.Archi_dir}')
            print(f'\t\t Weights_path               :{self.Weights_path}')
            print(f'\t\t Ckpt_Mod_Weights_fold      :{self.Ckpt_Mod_Weights_fold}')
            print(f'\t\t Final_Weights_fold         :{self.Final_Weights_fold}')
            print(f'\t\t     Hashing_mod_bn         :{self.Hashing_mod_bn}')
            print(f'\t\t     Hashing_model_path     :{self.Hashing_model_path}')
            print(f'\t\t     Hashing_weights_path   :{self.Hashing_weights_path}')
            print(f'\t\t     Training_mod_bn          :{self.Training_mod_bn}')
            print(f'\t\t     Training_model_path    :{self.Training_model_path}')
            print(f'\t\t     Training_weights_path  :{self.Training_weights_path}')
            print(f'\t\t Plot_path_dir              :{self.Plot_path_dir}')
            print(f'\t\t     Plot_path              :{self.Plot_path}')
        print('\t############ Model Parameters ############')
        if True:
            print(f'\t\t input_shape                :{self.input_shape}')
            print(f'\t\t output_shape               :{self.output_shape}')
            print(f'\t\t mdl_name                   :{self.mdl_name}')
            print(f'\t\t reg_mode                   :{self.reg_mode}')
            print(f'\t\t kwargs_reg                 :{self.kwargs_reg}')
        print('\t############ Hyper Parameters ############')
        if True:
            print(f'\t\t batch_size                 :{self.batch_size}')
            print(f'\t\t initial_epoch              :{self.initial_epoch}')
            print(f'\t\t epochs                     :{self.epochs}')
            print(f'\t\t steps_per_epoch            :{self.steps_per_epoch}')
            print(f'\t\t validation_steps           :{self.validation_steps}')
        print('\t############ Optimizer, Metrics Parameters ############')
        if True:
            print(f'\t\t opt_mode                   :{self.opt_mode}')
            print(f'\t\t opt_dict                   :{self.opt_dict}')
            print(f'\t\t compile_dict               :{self.compile_dict}')
        print('\t############ Callbacks ############')
        if True:
            print(f'\t\t Callbacks_list             :{self.Callbacks_list}')
            print(f'\t\t ReduceLROnPlateau_kwargs   :{self.ReduceLROnPlateau_kwargs}')
            print(f'\t\t CkptFold_det               :{self.CkptFold_det}')
            print(f'\t\t ModelCheckpoint_kwargs     :{self.ModelCheckpoint_kwargs}')
            print(f'\t\t csv_log_path               :{self.csv_log_path}')
        print('\t############ Inference Details ############')
        if True:
            print(f'\t\t expinf_dir                 :{self.expinf_dir}')
            print(f'\t\t expinf_wav_dir             :{self.expinf_wav_dir}')
        print('\t############ Results Details ############')
        if True:
            print(f'\t\t results_dir                :{self.results_dir}')
            print(f'\t\t results_vrs_dir            :{self.results_vrs_dir}')
        return ""

