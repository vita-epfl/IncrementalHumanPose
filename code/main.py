'''
Modified from:
https://github.com/meghshukla/ActiveLearningForHumanPose/tree/main
'''
import os
import copy
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau

from top_network import TopNetwork
from config import ParseConfig
from utils import visualize_image
from utils import heatmap_loss
from utils import count_parameters
from utils import calculate_fourier_response
from dataloader import load_hp_dataset
from dataloader import HumanPoseDataLoader
from evaluation import PercentageCorrectKeypoint
from StackedHourglass import PoseNet as Hourglass

# Global declarations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
logging.getLogger().setLevel(logging.INFO)
os.chdir(os.path.dirname(os.path.realpath(__file__)))


class Train(object):
    def __init__(self, pose_model, hyperparameters, dataset_obj, conf, tb_writer, task_no):
        """
        Class for training the human pose model

        :param pose_model: (torch.nn) Human pose model
        :param hyperparameters: (dict) Various hyperparameters used in training
        :param dataset_obj: (torch.utils.data.Dataset)
        :param conf: (Object of ParseConfig) Contains the configurations for the model
        :param tb_writer: (Object of SummaryWriter) Tensorboard writer to log values
        """

        self.conf = conf
        self.network = pose_model
        self.tb_writer = tb_writer
        self.dataset_obj = dataset_obj
        self.hyperparameters = hyperparameters
        self.task_no = task_no

        # Experiment Settings
        self.batch_size = conf.experiment_settings['batch_size']
        self.epoch = hyperparameters['num_epochs']
        self.optimizer = hyperparameters['optimizer']  # Adam / SGD
        self.loss_fn = hyperparameters['loss_fn']  # MSE
        self.learning_rate = hyperparameters['optimizer_config']['lr']
        self.start_epoch = hyperparameters['start_epoch']  # Used in case of resume training
        self.num_hm = conf.experiment_settings['num_hm']  # Number of heatmaps
        self.joint_names = self.dataset_obj.ind_to_jnt
        self.model_save_path = conf.model['save_path']

        if self.start_epoch > 0:
            self.epoch = self.start_epoch + hyperparameters['num_epochs_incr']


        # Stacked Hourglass scheduling
        min_lr = 0.000003

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5, cooldown=2, min_lr=min_lr, verbose=True)

        self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, self.batch_size,
                                                            shuffle=True, num_workers=2, drop_last=True)


    def train_model(self):
        """
        Training loop
        :return:
        """

        print("Initializing training: Epochs - {}\tBatch Size - {}".format(
            self.hyperparameters['num_epochs'], self.batch_size))

        best_val_pose = np.inf
        best_epoch_pose = -1
        global_step = 0

        # Variable to store all the loss values for logging
        loss_across_epochs = []
        validation_across_epochs = []

        for e in range(self.start_epoch, self.epoch):
            epoch_loss = []

            # Network alternates between train() and validate()
            self.network.train()

            self.dataset_obj.input_dataset(train=True)

            # Training loop
            logging.info('Training for epoch: {}'.format(e+1))
            for (images, heatmaps, _, _, _, gt_per_image, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):

                assert split[0] == 0, "Training split should be 0."

                self.optimizer.zero_grad()
                outputs = self.network(images)          # images.cuda() done internally within the model
                loss = heatmap_loss(outputs[self.task_no], heatmaps, loss_fn=self.loss_fn)                 # heatmaps transferred to GPU within the function

                loss = torch.mean(loss)

                loss.backward()
                if self.conf.tensorboard:
                    self.tb_writer.add_scalar('Train/Loss_batch', torch.mean(loss), global_step)
                epoch_loss.append(loss.item())

                # Weight update
                self.optimizer.step()
                global_step += 1


            # Epoch training ends -------------------------------------------------------------------------------------
            epoch_loss = np.mean(epoch_loss)

            validation_loss_pose = self.validation(e)

            # Learning rate scheduler on the Human Pose validation loss
            self.scheduler.step(validation_loss_pose)

            # TensorBoard Summaries
            if self.conf.tensorboard:
                self.tb_writer.add_scalar('Train', torch.tensor([epoch_loss]), global_step)
                self.tb_writer.add_scalar('Validation/HG_Loss', torch.tensor([validation_loss_pose]), global_step)

            # Save if best model
            if best_val_pose > validation_loss_pose:
                torch.save(self.network.state_dict(),
                           os.path.join(self.model_save_path, f'model_checkpoints/pose_net_{self.task_no}.pth'))

                best_val_pose = validation_loss_pose
                best_epoch_pose = e + 1

                torch.save({'epoch': e + 1,
                            'optimizer_load_state_dict': self.optimizer.state_dict(),
                            'mean_loss_train': epoch_loss,
                            'mean_loss_validation': {'Pose': validation_loss_pose}},
                           os.path.join(self.model_save_path, f'model_checkpoints/optim_best_model_{self.task_no}.tar'))


            print("Loss at epoch {}/{}: (train:Pose) {}\t"
                  "(validation:Pose) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epoch,
                epoch_loss,
                validation_loss_pose,
                best_epoch_pose))

            loss_across_epochs.append(epoch_loss)
            validation_across_epochs.append(validation_loss_pose)

            # Save the loss values
            f = open(os.path.join(self.model_save_path, f'model_checkpoints/loss_data_{self.task_no}.txt'), "w")
            f_ = open(os.path.join(self.model_save_path, f'model_checkpoints/validation_data_{self.task_no}.txt'), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f_.write("\n".join([str(lsx) for lsx in validation_across_epochs]))
            f.close()
            f_.close()

        if self.conf.tensorboard:
            self.tb_writer.close()
        logging.info("Model training completed\nBest validation loss (Pose): {}\tBest Epoch: {}".format(best_val_pose, best_epoch_pose))


    def validation(self, e):
        """

        :param e: Epoch
        :return:
        """
        with torch.no_grad():
            # Stores the loss for all batches
            epoch_val_pose = []
            self.network.eval()

            # Augmentation only needed in Training
            self.dataset_obj.input_dataset(validate=True)

            # Compute and store batch-wise validation loss in a list
            logging.info('Validation for epoch: {}'.format(e+1))
            for (images, heatmaps, _, _, _, gt_per_img, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):

                assert split[0] == 1, "Validation split should be 1."

                outputs = self.network(images)
                loss_val_pose = heatmap_loss(outputs[self.task_no], heatmaps)

                loss_val_pose = torch.mean(loss_val_pose)
                epoch_val_pose.append(loss_val_pose.item())

            return np.mean(epoch_val_pose)


class Metric(object):
    def __init__(self, network, dataset_obj, conf, task_no, csv_name=""):
        '''
        Class for Testing the model:
            1. Compute ground truth and predictions
            2. Computing metrics: PCK@0.x
        :param network: (torch.nn) Hourglass network to compute predictions
        :param dataset_obj: (Dataset object) Handles data to be fed to PyTorch DataLoader
        :param conf: (Object of ParseConfig) Configuration for the experiment
        '''

        self.dataset_obj = dataset_obj
        self.dataset_obj.input_dataset(validate=True)

        self.network = network
        self.viz=conf.viz                                                                       # Controls visualization
        self.conf = conf
        self.batch_size = conf.experiment_settings['batch_size']
        self.ind_to_jnt = self.dataset_obj.ind_to_jnt
        self.csv_name = csv_name
        self.task_no = task_no


        self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, batch_size=self.batch_size,
                                                            shuffle=False, num_workers=2)


    def inference(self):
        '''
        Obtains model inference
        :return: None
        '''

        self.network.eval()
        logging.info("Starting model inference")

        outputs_ = None
        scale_ = None
        num_gt_ = None
        dataset_ = None
        name_ = None
        gt_ = None
        normalizer_ = None

        with torch.no_grad():
            for (images, _, gt, name, dataset, num_gt, split, _, scale_params, normalizer, joint_exist) in tqdm(
                    self.torch_dataloader):

                assert split[0] == 1, "Validation split should be 1."
                outputs = self.network(images)
                outputs = outputs[self.task_no]
                outputs = outputs[:, -1]

                try:
                    outputs_ = torch.cat((outputs_, outputs.cpu().clone()), dim=0)
                    scale_['scale_factor'] = torch.cat((scale_['scale_factor'], scale_params['scale_factor']), dim=0)
                    scale_['padding_u'] = torch.cat((scale_['padding_u'], scale_params['padding_u']), dim=0)
                    scale_['padding_v'] = torch.cat((scale_['padding_v'], scale_params['padding_v']), dim=0)
                    num_gt_ = torch.cat((num_gt_, num_gt), dim=0)
                    dataset_ = dataset_ + dataset
                    name_ = name_ + name
                    gt_ = torch.cat((gt_, gt), dim=0)
                    normalizer_ = torch.cat((normalizer_, normalizer), dim=0)

                except TypeError:
                    outputs_ = outputs.cpu().clone()
                    scale_ = copy.deepcopy(scale_params)
                    num_gt_ = num_gt
                    dataset_ = dataset
                    name_ = name
                    gt_ = gt
                    normalizer_ = normalizer

                # Generate visualizations (256x256) for that batch of images
                if self.conf.viz:
                    hm_uv_stack = []
                    # Compute u,v values from heatmap
                    for i in range(images.shape[0]):
                        hm_uv = self.dataset_obj.estimate_uv(hm_array=outputs.cpu().numpy()[i],
                                                             pred_placeholder=-np.ones_like(gt[i].numpy()))
                        hm_uv_stack.append(hm_uv)
                    hm_uv = np.stack(hm_uv_stack, axis=0)
                    self.visualize_predictions(image=images.numpy(), name=name, dataset=dataset, gt=gt.numpy(), pred=hm_uv)


        scale_['scale_factor'] = scale_['scale_factor'].numpy()
        scale_['padding_u'] = scale_['padding_u'].numpy()
        scale_['padding_v'] = scale_['padding_v'].numpy()

        model_inference = {'heatmap': outputs_.numpy(), 'scale': scale_, 'dataset': dataset_,
                           'name': name_, 'gt': gt_.numpy(), 'normalizer': normalizer_.numpy()}

        return model_inference


    def keypoint(self, infer):
        '''

        :param infer:
        :return:
        '''

        heatmap = infer['heatmap']
        scale = infer['scale']
        dataset = infer['dataset']
        name = infer['name']
        gt = infer['gt']
        normalizer = infer['normalizer']

        hm_uv_stack = []

        csv_columns = ['name', 'dataset', 'normalizer', 'joint', 'uv']

        gt_csv = []
        pred_csv = []

        # Iterate over all heatmaps to obtain predictions
        for i in range(gt.shape[0]):

            heatmap_ = heatmap[i]

            gt_uv = gt[i]
            hm_uv = self.dataset_obj.estimate_uv(hm_array=heatmap_, pred_placeholder=-np.ones_like(gt_uv))
            hm_uv_stack.append(hm_uv)

            # Scaling the point ensures that the distance between gt and pred is same as the scale of normalization
            scale_factor = scale['scale_factor'][i]
            padding_u = scale['padding_u'][i]
            padding_v = scale['padding_v'][i]

            # Scaling ground truth
            gt_uv_correct = np.copy(gt_uv)
            hm_uv_correct = np.copy(hm_uv)

            gt_uv_correct[:, :, 1] -= padding_v
            gt_uv_correct[:, :, 0] -= padding_u
            gt_uv_correct /= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)

            # Scaling predictions
            hm_uv_correct[:, :, 1] -= padding_v
            hm_uv_correct[:, :, 0] -= padding_u
            hm_uv_correct /= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)

            assert gt_uv_correct.shape == hm_uv_correct.shape, "Mismatch in gt ({}) and prediction ({}) shape".format(
                gt_uv_correct.shape, hm_uv_correct.shape)

            # Iterate over joints
            for jnt in range(14):
                gt_entry = {
                    'name': name[i],
                    'dataset': dataset[i],
                    'normalizer': normalizer[i],
                    'joint': self.ind_to_jnt[jnt],
                    'uv': gt_uv_correct[:, jnt, :].astype(np.float32)
                }

                pred_entry = {
                    'name': name[i],
                    'dataset': dataset[i],
                    'normalizer': normalizer[i],
                    'joint': self.ind_to_jnt[jnt],
                    'uv': hm_uv_correct[:, jnt, :].astype(np.float32)
                }

                gt_csv.append(gt_entry)
                pred_csv.append(pred_entry)


        pred_csv = pd.DataFrame(pred_csv, columns=csv_columns)
        gt_csv = pd.DataFrame(gt_csv, columns=csv_columns)

        pred_csv.sort_values(by='dataset', ascending=True, inplace=True)
        gt_csv.sort_values(by='dataset', ascending=True, inplace=True)

        assert len(pred_csv.index) == len(gt_csv.index), "Mismatch in number of entries in pred and gt dataframes."

        pred_csv.to_csv(os.path.join(self.conf.model['save_path'], f'model_checkpoints/pred_{self.csv_name}.csv'), index=False)
        gt_csv.to_csv(os.path.join(self.conf.model['save_path'], f'model_checkpoints/gt_{self.csv_name}.csv'), index=False)
        logging.info('Pandas dataframe saved successfully.')

        return gt_csv, pred_csv


    def visualize_predictions(self, image=None, name=None, dataset=None, gt=None, pred=None):

        dataset_viz = {}
        dataset_viz['img'] = image
        dataset_viz['name'] = name
        dataset_viz['display_string'] = name
        dataset_viz['split'] = np.ones(image.shape[0])
        dataset_viz['dataset'] = dataset
        dataset_viz['bbox_coords'] = np.zeros([image.shape[0], 4, 4])
        dataset_viz['num_persons'] = np.ones([image.shape[0], 1])
        dataset_viz['gt'] = gt
        dataset_viz['pred'] = pred

        dataset_viz = self.dataset_obj.recreate_images(gt=True, pred=True, external=True, ext_data=dataset_viz)
        visualize_image(dataset_viz, save_dir=self.conf.model['save_path'], bbox=False)


    def compute_metrics(self, gt_df=None, pred_df=None):
        '''
        Loads the ground truth and prediction CSVs into memory.
        Evaluates Precision, FPFN metrics for the prediction and stores them into memory.
        :return: None
        '''

        # Ensure that same datasets have been loaded
        assert all(pred_df['dataset'].unique() == gt_df['dataset'].unique()), \
            "Mismatch in dataset column for gt and pred"

        logging.info('Generating evaluation metrics for dataset:')
        # Iterate over unique datasets
        for dataset_ in gt_df['dataset'].unique():
            logging.info(str(dataset_))

            # Separate out images based on dataset
            pred_ = pred_df.loc[pred_df['dataset'] == dataset_]
            gt_ = gt_df.loc[gt_df['dataset'] == dataset_]

            # Compute scores
            pck_df = PercentageCorrectKeypoint(
                pred_df=pred_, gt_df=gt_, config=self.conf, jnts=list(self.ind_to_jnt.values()))

            # Save the tables
            if dataset_ == 'mpii':
                metric_ = 'PCKh'
            else:
                metric_ = 'PCK'

            pck_df.to_csv(os.path.join(self.conf.model['save_path'],
                                       'model_checkpoints/{}_{}_{}.csv'.format(metric_, dataset_, self.csv_name)),
                          index=False)

        print("Metrics computation completed.")


    def eval(self):
        '''

        :return:
        '''
        model_inference = self.inference()
        gt_csv, pred_csv = self.keypoint(model_inference)
        self.compute_metrics(gt_df=gt_csv, pred_df=pred_csv)



def load_models(conf, load_pose, model_dir, task_no):
    """

    :param conf:
    :param load_pose:
    :param model_dir:
    :return:
    """

    # Initialize Hourglass
    # Elsewhere, resume training ensures the code creates a copy of the best models from the interrupted run.


    logging.info('Initializing Hourglass Network')
    pose_net = Hourglass(arch=conf.architecture['hourglass'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pose_net = TopNetwork(pose_net, device)
    pose_net.add_head() # for the base dataset
    pose_net.add_head() # for the incremental dataset
    print('Number of parameters (Hourglass): {}\n'.format(count_parameters(pose_net)))

    # Load Pose model (code is independent of architecture)
    if load_pose:

        # Load model
        logging.info('Loading Pose model from: ' + model_dir)
        pose_net.load_state_dict(torch.load(os.path.join(model_dir, f'model_checkpoints/pose_net_{task_no}.pth'), map_location='cpu'))
        logging.info("Successfully loaded Pose model.")

        if conf.resume_training:
            logging.info('\n-------------- Resuming training (Loading PoseNet) --------------\n')
            torch.save(pose_net.state_dict(), os.path.join(conf.model['save_path'], f'model_checkpoints/pose_net_{task_no}.pth'))

    # CUDA support: Single/Multi-GPU
    # Hourglass net has CUDA definitions inside __init__()
    logging.info('Successful: Model transferred to GPUs.\n')

    return pose_net


def define_hyperparams(conf, pose_model,task_no, loss_fn=None):#(conf, net, learnloss):
    """

    :param conf:
    :param pose_model:
    :return:
    """
    logging.info('Initializing the hyperparameters for the experiment.')
    hyperparameters = dict()
    hyperparameters['optimizer_config'] = {
                                           'lr': conf.experiment_settings['lr'],
                                           'weight_decay': conf.experiment_settings['weight_decay']
                                          }
    hyperparameters['loss_params'] = {'size_average': True}
    hyperparameters['num_epochs'] = conf.experiment_settings['epochs']
    hyperparameters['num_epochs_incr'] = conf.experiment_settings['epochs_incr']
    hyperparameters['start_epoch'] = 0  # Used for resume training

    # Parameters declared to the optimizer
    logging.info('Parameters of PoseNet passed to Optimizer')
    params_list = [{'params': pose_model.parameters()}]

    hyperparameters['optimizer'] = torch.optim.Adam(params_list, **hyperparameters['optimizer_config'])

    if conf.resume_training:
        logging.info('Loading optimizer state dictionary')
        optim_dict = torch.load(os.path.join(conf.model['base_load_path'], f'model_checkpoints/optim_best_model_{task_no}.tar'))

        hyperparameters['optimizer'].load_state_dict(optim_dict['optimizer_load_state_dict'])
        logging.info('Optimizer state loaded successfully.\n')

        logging.info('Optimizer and Training parameters:\n')
        for key in optim_dict:
            if key == 'optimizer_load_state_dict':
                logging.info('Param group length: {}'.format(len(optim_dict[key]['param_groups'])))
            else:
                logging.info('Key: {}\tValue: {}'.format(key, optim_dict[key]))

        logging.info('\n')
        hyperparameters['start_epoch'] = optim_dict['epoch']
        hyperparameters['mean_loss_validation'] = optim_dict['mean_loss_validation']
    '''
    if loss_fn is None:
        hyperparameters['loss_fn'] = torch.nn.MSELoss(reduction='none')
    else:
        hyperparameters['loss_fn'] = loss_fn
    '''
    hyperparameters['loss_fn'] = loss_fn
    return hyperparameters


def main():
    """
    Control flow for the code
    """

    # 1. Load configuration file --------------------------------------------------------------------------------------
    logging.info('Loading configurations.\n')
    conf  = ParseConfig()
    task_no = 0


    # 2. Loading datasets ---------------------------------------------------------------------------------------------
    logging.info('Loading pose dataset(s)\n')
    dataset_dict_base = load_hp_dataset(dataset_conf=conf.dataset, model_conf=conf.model)
    dataset_dict_incr = load_hp_dataset(dataset_conf=conf.dataset_incr, model_conf=conf.model)


    # 3. Defining the network -----------------------------------------------------------------------------------------
    logging.info('Initializing (and loading) human pose network\n')
    pose_model = load_models(conf=conf, load_pose=conf.model['load'],
                                      model_dir=conf.model['base_load_path'], task_no=task_no)


    # 4. Defining DataLoader -------------------------------------------------------------------------------------------
    logging.info('Defining DataLoader.\n')
    datasets_base = HumanPoseDataLoader(dataset_dict=dataset_dict_base, conf=conf, use_incr=False)
    datasets_incr = HumanPoseDataLoader(dataset_dict=dataset_dict_incr, conf=conf, use_incr=True)


    # 4.a: Delete models to remove stray computational graphs (esp. for EGL)

    del pose_model
    torch.cuda.empty_cache()
    
    logging.info('Re-Initializing (and loading) human pose network.\n')
    pose_model = load_models(conf=conf, load_pose=conf.model['load'],
                                      model_dir=conf.model['base_load_path'], task_no=task_no)


    # 6. Defining Hyperparameters, TensorBoard directory ---------------------------------------------------------------
    logging.info('Initializing experiment settings.')
    hyperparameters = define_hyperparams(conf=conf, pose_model=pose_model, task_no=task_no)

    if conf.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(conf.model['save_path'], 'tensorboard'))
    else:
        writer = None


    # 7. Train the model

    '''
    if conf.train:
        train_obj = Train(pose_model=pose_model, hyperparameters=hyperparameters,
                          dataset_obj=datasets_base, conf=conf, tb_writer=writer, task_no=task_no)
        train_obj.train_model()

        del train_obj
        # Reload the best model for metric evaluation
        conf.resume_training = False
        pose_model = load_models(conf=conf, load_pose=True, model_dir=conf.model['save_path'], task_no=task_no)

    # Metrics before incremental learning
    if conf.metric:
        metric_obj_base = Metric(network=pose_model, dataset_obj=datasets_base, conf=conf, task_no=0, csv_name="base_before")
        metric_obj_base.eval()

        metric_obj_incr = Metric(network=pose_model, dataset_obj=datasets_incr, conf=conf, task_no=1, csv_name="incr_before")
        metric_obj_incr.eval()
    '''

    # Increment
    conf.resume_training = True
    pose_model = load_models(conf=conf, load_pose=True, model_dir=conf.model['base_load_path'], task_no=task_no)
    reg_lambda = conf.reg_lambda
    incr_loss_fn = None # default loss fn

    # Choose an incremental strategy
    if conf.increment_strategy == "finetune":
        # Freeze all parameters except the heads
        for param in pose_model.parameters():
            param.requires_grad = False

        for head in pose_model.heads:
            for param in head.parameters():
                param.requires_grad = True
                break

    if conf.increment_strategy == "reg_base":
        # get base parameters
        base_param = []
        for param in pose_model.parameters():
            base_param.append(param.clone())

        def regularized_loss(output, target):
            loss = ((output - target) ** 2).mean(dim=[1, 2, 3]) # MSE
            sum = 0
            # Regularize the model by discouraging deviation from base model's parameters
            for p_old, p_new in zip(base_param, pose_model.parameters()):
                sum += (torch.linalg.norm(p_new.flatten(), 2) - torch.linalg.norm(p_old.flatten(), 2)) ** 2
            loss += reg_lambda * sum
            return loss

        incr_loss_fn = regularized_loss

    if conf.increment_strategy == "reg_freq":
        # Get base model's conv layers' Fourier response
        fourier_response_base = calculate_fourier_response(pose_model)
        def fourier_loss(output, target):
            loss = ((output - target) ** 2).mean(dim=[1, 2, 3]) # MSE
            sum = 0
            fourier_response_new = calculate_fourier_response(pose_model)
            # Regularize the model by discouraging deviation from base model's Fourier response
            for r_old, r_new in zip(fourier_response_base, fourier_response_new):
                sum += torch.mean((r_new - r_old) ** 2)
            loss += reg_lambda * sum
            return loss

        incr_loss_fn = fourier_loss

    # Define hyperparameters, loss function
    hyperparameters_incr = define_hyperparams(conf=conf, pose_model=pose_model, task_no=task_no,
                                                  loss_fn=incr_loss_fn)
    task_no = task_no + 1
    # Train over the base model
    train_obj = Train(pose_model=pose_model, hyperparameters=hyperparameters_incr,
                      dataset_obj=datasets_incr, conf=conf, tb_writer=writer, task_no=task_no)
    train_obj.train_model()
    del train_obj

    conf.resume_training = False
    pose_model = load_models(conf=conf, load_pose=True,  model_dir=conf.model['incr_load_path'], task_no=task_no)

    # Metrics after incremental learning
    if conf.metric:
        metric_obj_base = Metric(network=pose_model, dataset_obj=datasets_base, conf=conf, task_no=0, csv_name="base_after")
        metric_obj_base.eval()

        metric_obj_incr = Metric(network=pose_model, dataset_obj=datasets_incr, conf=conf, task_no=1, csv_name="incr_after")
        metric_obj_incr.eval()


if __name__ == "__main__":
    main()
