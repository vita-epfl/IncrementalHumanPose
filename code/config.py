import os
import yaml
import shutil
import logging
from pathlib import Path


class ParseConfig(object):
    """
    Loads and returns the configuration specified in configuration.yml
    """
    def __init__(self):


        # 1. Load the configuration file ------------------------------------------------------------------------------
        try:
            f = open('configuration.yml', 'r')
            conf_yml = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        except FileNotFoundError:
            logging.warning('Could not find configuration.yml')
            exit()

        # 2. Initializing ParseConfig object --------------------------------------------------------------------------
        self.model = conf_yml['model']
        self.dataset = conf_yml['dataset']
        self.dataset_incr = conf_yml['dataset_incr']
        self.num_images = conf_yml['num_images']
        self.train = conf_yml['train']
        self.metric = conf_yml['metric']
        self.experiment_settings = conf_yml['experiment_settings']
        self.architecture = conf_yml['architecture']
        self.viz = conf_yml['visualize']
        self.tensorboard = conf_yml['tensorboard']
        self.resume_training = conf_yml['resume_training']
        self.increment_strategy = conf_yml['increment_strategy']
        self.reg_lambda = conf_yml['reg_lambda']


        # 3. Extra initializations based on configuration chosen ------------------------------------------------------

        # Number of heatmaps (or joints) based on the dataset
        if self.dataset['load'] == 'mpii':
            self.experiment_settings['num_hm'] = 16
            self.architecture['hourglass']['num_hm'] = 16

        else:
            assert self.dataset['load'] == 'lsp' or self.dataset['load'] == 'merged',\
                "num_hm defined only for 'mpii' and 'lsp' datasets"
            self.experiment_settings['num_hm'] = 14
            self.architecture['hourglass']['num_hm'] = 14


        # Only Hourglass supported
        assert self.model['type'] in ['hourglass'], "Invalid Model type given: {}.".format(self.model['type'])

        # Resume training
        if self.resume_training:
            assert self.model['load'], "Resume training specified but model load == False"
            assert self.train, "Resume training requires train to be True."

        if self.experiment_settings['all_joints']:
            assert self.experiment_settings['occlusion'], "Occlusion needs to be true if all joints is true."


        # 4. Create directory for model save path ----------------------------------------------------------------------
        self.experiment_name = conf_yml['experiment_name']
        i = 1
        model_save_path = os.path.join(self.model['save_path'], self.experiment_name + '_' + str(i))
        while os.path.exists(model_save_path):
            i += 1
            model_save_path = os.path.join(self.model['save_path'], self.experiment_name + '_' + str(i))

        logging.info('Saving the model at: ' + model_save_path)
        os.makedirs(os.path.join(model_save_path, 'model_checkpoints'))

        # Copy the configuration file into the model dump path
        code_directory = Path(os.path.abspath(__file__)).parent
        shutil.copytree(src=str(code_directory),
                        dst=os.path.join(model_save_path, code_directory.parts[-1]))

        self.model['save_path'] = model_save_path
        self.model['incr_load_path'] = model_save_path