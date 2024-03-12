import argparse
import os
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json
from data.load_data import loadingData
from data.load_test_data import loadingTestData
from BlockModule import DeepShare2
from basicModule import *
import scipy.io
# loss
from Loss import HybridLoss, CrossEntropy2d, L1_Charbonnier_loss
from metrics import quality_assessment
from torch.autograd import Variable

# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = ''


def main():
    # parsers UseUnLabeledMixUp
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False, default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, required=True, help="batch size, default set to 64")
    train_parser.add_argument("--epochs", type=int, default=20,  help="epochs, default set to 20")
    train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    train_parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--train_dir_mslabel", type=str, required=True, help="directory of train spectral dataset")
    train_parser.add_argument("--eval_dir_ms", type=str, help="directory of evaluation spectral dataset")
    train_parser.add_argument("--test_dir", type=str, required=True, help="directory of test spectral dataset")
    train_parser.add_argument("--data_train_num", type=int, required=True, help="how many .mat files used in each epoch")
    train_parser.add_argument("--data_eval_num", type=int, help="how many .mat files used in each epoch")
    train_parser.add_argument("--data_test_num", type=int, required=True, help="how many .mat files used in each epoch")
    train_parser.add_argument("--model_title", type=str, default="DeepShare",
                              help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-03,              #1e-3
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 7)")


    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False, default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="7", help="gpu ids (default: 7)")
    test_parser.add_argument("--test_dir", type=str, required=True, help="directory of test spectral dataset")
    test_parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")
    test_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    test_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    test_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    test_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    test_parser.add_argument("--n_colors", type=int, required=True, help="n_colors, default set to 31")
    test_parser.add_argument("--n_scale", type=int, default=8, help="n_scale, default set to 2")
    test_parser.add_argument("--model_title", type=str, default="DeepShare",
                              help="model_title, default set to model_title")
    test_parser.add_argument("--result_path", type=str, default="./Result",
                             help="result_path, directory of result")
    test_parser.add_argument("--data_test_num", type=int, required=True, help="how many .mat files used in each epoch")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass


bce_loss = torch.nn.BCEWithLogitsLoss()




def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    #load conversion_matrix and by multiplying the conversion matrix with images, we can get images with 8 channels

    print('===> Loading datasets')

    train_mslabel_set = loadingData(image_dir=args.train_dir_mslabel, augment=True, total_num=args.data_train_num)

    if args.dataset_name == 'Cave':
        colors = 31
    elif args.dataset_name == 'NTIRE2020':
        colors = 31
    elif args.dataset_name == 'Harvard':
        colors = 31
    else:
        colors = 128

    print('===> Building model')
    if args.model_title =="DeepShare":
        net = DeepShare2(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=colors, n_blocks=args.n_blocks, n_feats=args.n_feats,
                n_scale=args.n_scale, res_scale=0.1, use_share=args.use_share, conv=default_conv)


    model_title = args.dataset_name + "_" + args.model_title + '_Blocks=' + str(args.n_blocks) + '_Subs' + str(
        args.n_subs) + '_Ovls' + str(args.n_ovls) + '_Feats=' + str(args.n_feats)
    model_name = "/home/featurize/work/HSISR-main/checkpoints_pl/CAVE/x8/GDBN/Cave_Cave_DeepShare_Blocks=3_Subs8_Ovls2_Feats=1240_ckpt_epoch_29.pth"
    args.model_title = model_title

    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
            #state_dict = torch.load(model_name)
            #net.load_state_dict(state_dict, strict=False)

        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()
    print("testing_device ****", device)


    # loss functions to choose
    mse_loss = torch.nn.MSELoss()
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    L1_loss = L1_Charbonnier_loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.25,
                                                           patience=0, verbose=False,                       #patience=2/1
                                                           threshold=0.1, threshold_mode='abs',
                                                           cooldown=0, min_lr=0, eps=1e-08)
    epoch_meter_mslabel = meter.AverageValueMeter()

    writer = SummaryWriter('runs/' + model_title + '_' + str(time.ctime()))



    print('===> Start training')
    Lr_change = 0
    for e in range(start_epoch, args.epochs):

        epoch_meter_mslabel.reset()
        print("Start epoch {}, labeled ms learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))

        iteration = 0
        train_mslabel_loader = DataLoader(train_mslabel_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
        train_mslabel_iter = iter(train_mslabel_loader)
        i = 0
        for batch_mslabel in train_mslabel_iter:
            # training for spectral images


            x, lms, gt = batch_mslabel
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()
            y_ms_l = net(x, lms, modality="spectral")
            loss = h_loss(y_ms_l, gt)
            epoch_meter_mslabel.add(loss.item())
            loss.backward()
            optimizer.step()
            
            # if(iteration==3200):
            #     output = []
            #     for i in range(args.batch_size):
            #         output.append(y_ms_l[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
            #     save_dir = model_title + '_' + str(time.ctime()) + '.npy'
            #     np.save(save_dir, output)
            #     print("Test finished, test results saved to .npy file at ", save_dir)


        # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): ms Loss: {:.6f}".format(time.ctime(),
                                                                                            args.n_blocks,
                                                                                            args.n_subs,
                                                                                            args.n_feats,
                                                                                            args.gpus, e + 1,
                                                                                            iteration + 1,
                                                                                            len(train_mslabel_loader),
                                                                                            loss.item()))
                n_iter = e * len(train_mslabel_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss_ms', loss, n_iter)
                
            iteration += 1
            Lr_change += 1

        print("===> {}\tEpoch {} Training mslabel Complete: Avg. Loss: {:.6f}".format(time.ctime(), e + 1,
                                                                              epoch_meter_mslabel.value()[0]))

        # run validation set every epoch
        print("Running testset")
        print('===> Loading testset')
        test_set = loadingTestData(image_dir=args.test_dir, augment=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        MPSNR = validate(args, test_loader, net, model_title, e)
        scheduler.step(MPSNR)
        print("lr is changed to or stay in: ", optimizer.param_groups[0]["lr"])
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss_mslabel', epoch_meter_mslabel.value()[0], iteration)
        # save model weights at checkpoints every 10 epochs
        # save_checkpoint(args, net, e + 1) 

    print("\nDone, train complete！ ")
    ## Save the testing results
    print("Running testset")
    print('===> Loading testset')
    test_set = loadingTestData(image_dir=args.test_dir, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net.eval().cuda()
    with torch.no_grad():
        output = []
        test_number = 0
        for i, (x, lms, gt) in enumerate(test_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            y = net(x, lms, modality="spectral")
            y, gt, lms = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0), lms.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    print(indices)
    save_dir = model_title + '.npy'
    # np.save(save_dir, output)

def sum_dict(a, b):
    temp = dict()
    for key in a.keys() | b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


def adjust_learning_rate(start_lr, optimizer, iters, total_iter):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    # lr = start_lr * (0.1 ** (epoch // 30))
    lr = start_lr * (0.8 ** (iters // (total_iter/3)))
    print("lr has changed to :", lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_D(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    # lr = start_lr * (0.1 ** (epoch // 30))
    lr = start_lr * (0.3 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, model_title, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        output = []
        test_number = 0
        for i, (x, lms, gt) in enumerate(loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            y = model(x, lms, modality="spectral")
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

        # save_dir = "/data/test.npy"
    # save_dir = model_title + '_' + str(epoch) +'.npy'
    # np.save(save_dir, output)
    # print("Test finished, test results saved to .npy file at ", save_dir)
    print("Test finished:____WOW____WOW______")
    print(indices)
    QIstr = args.model_title + '_No_SPN2_' + str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))
    # back to training mode
    model.train()
    return indices['MPSNR']


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    test_set = loadingTestData(image_dir=args.test_dir, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():
        # epoch_meter = meter.AverageValueMeter()
        # epoch_meter.reset()
        # loading model
        model = DeepShare2(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=args.n_colors, n_blocks=args.n_blocks, n_feats=args.n_feats,
                      n_scale=args.n_scale, res_scale=0.1, use_share=True, conv=default_conv)

        state_dict = torch.load(args.model_dir)
        model.load_state_dict(state_dict, strict=False)
        #checkpoint = torch.load(args.model_dir)
        #model.load_state_dict(checkpoint["model"].state_dict())

        model.to(device).eval()
        mse_loss = torch.nn.MSELoss()
        output = []
        test_number = 0
        for i, (x, lms, gt) in enumerate(test_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            y = model(x, lms, modality="spectral")

            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0], :gt.shape[1], :]
            if i == 0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    save_dir = args.result_path + args.model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = args.model_title + '_NoCa_' + str(time.ctime()) + ".txt"
    json.dump(indices, open(QIstr, 'w'))


def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints_pl/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()
