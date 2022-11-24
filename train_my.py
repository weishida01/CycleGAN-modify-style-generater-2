"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
import os
from data import create_dataset
from models import create_model
from my_utils.base_dataset import save_results
import tqdm
from options.base_options import BaseOptions




class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # python train_my.py --dataroot /home/weishida/dataset/cycle_gan
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--save_results_freq', type=int, default=20,help='frequency of showing training results on screen')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_log_freq', type=int, default=20, help='frequency of showing training results on console')

        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,help='number of epochs to linearly decay learning rate to zero')


        # python train_my.py --dataroot /home/weishida/dataset/cycle_gan --epoch 30 --epoch_count 31 --continue_train
        # python train_my.py --dataroot /home/weishida/dataset/cycle_gan --epoch 21 --continue_train --load_dir /home/weishida/code/CycleGan/CycleGAN_5_modify_stytle_c1_inpaint2/checkpoints/distribution_cyclegan_2
        # python train_my.py --dataroot /home/weishida/dataset/cycle_gan --epoch 33 --epoch_count 34 --continue_train --load_dir /home/weishida/code/CycleGan/CycleGAN_5_modify_stytle_c1_inpaint2/checkpoints/distribution_cyclegan

        parser.add_argument('--epoch_count', type=int, default=1,help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_dir', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')


        parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='distribution_cyclegan',help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='cycle_gan',help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')


        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser






if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)


    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        pbar = tqdm.tqdm(total=dataset_size, leave=True, desc='train', dynamic_ncols=True)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.save_results_freq == 0:   # save images
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                if not os.path.exists(opt.checkpoints_dir):
                    os.makedirs(opt.checkpoints_dir)
                result_dir = os.path.join(opt.checkpoints_dir, opt.name)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                result_dir = os.path.join(result_dir, 'results')
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                result_dir = os.path.join(result_dir, 'epoch_{}'.format(epoch))
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                result_dir = os.path.join(result_dir, 'total_iters_{}'.format(total_iters))
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                save_results(model.get_current_visuals(),result_dir)


            losses = model.get_current_losses()
            message = '(epoch:%d,iters:%d) ' % (epoch, epoch_iter)
            for k, v in losses.items():
                message += '%s:%.3f ' % (k, v)

            if total_iters % opt.save_log_freq == 0:  #save logging information to the disk
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)  # save the message


            pbar.update(opt.batch_size)  # 手动更新迭代 更新步长1
            pbar.set_postfix(dict(ms=message))  # 增加显示参数    print training losses
        pbar.close()


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
