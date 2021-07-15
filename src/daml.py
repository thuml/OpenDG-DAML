import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')

import dalib.vision.datasets as datasets

from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.lr_scheduler import StepwiseLR, StepLR
import numpy as np
import os
from tools.daml_utils import resnet18_fast, ClassifierFast

RG = np.random.default_rng()

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    val_save_dir = 'runs/' + args.savename + '_best_val.tar'
    test_save_dir = 'runs/' + args.savename + '_best_test.tar'
    final_save_dir = 'runs/' + args.savename + '_final.tar'

    # Data loading code

    if args.data == 'PACS':

        n1 = 0
        n2 = 1
        n3 = 1
        n4 = 0
        n5 = 1
        n6 = 1
        n7 = 1

        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif args.data == 'OfficeHome':

        n1 = 3
        n2 = 6
        n3 = 11
        n4 = 1
        n5 = 2
        n6 = 3
        n7 = 11

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    num_source_domain = len(args.source)

    # Open-Domain Split
    S123 = [i for i in range(n1)]
    S12 = [i + n1 for i in range(n2)]
    S13 = [i + n1 + n2 for i in range(n2)]
    S23 = [i + n1 + n2 * 2 for i in range(n2)]
    S1 = [i + n1 + n2 * 3 for i in range(n3)]
    S2 = [i + n1 + n2 * 3 + n3 for i in range(n3)]
    S3 = [i + n1 + n2 * 3 + n3 * 2 for i in range(n3)]

    ST1 = [S123[i] for i in range(n4)] \
        + [S12[i] for i in range(n5)] + [S13[i] for i in range(n5)] + [S23[i] for i in range(n5)] \
        + [S1[i] for i in range(n6)] + [S2[i] for i in range(n6)] + [S3[i] for i in range(n6)]
    T1 = [i + n1 + n2 * 3 + n3 * 3 for i in range(n7)]

    source_classes = [[], [], []]
    source_classes[0] = S1 + S12 + S13 + S123
    source_classes[1] = S2 + S12 + S23 + S123
    source_classes[2] = S3 + S13 + S23 + S123
    target_classes = ST1 + T1

    print(source_classes[0])
    print(source_classes[1])
    print(source_classes[2])
    print(target_classes)
    print(ST1)

    dataset = datasets.__dict__[args.data]

    train_source_iter_list = []
    for j, the_source in enumerate(args.source):
        train_source_dataset = dataset(root=args.root, task=the_source, filter_class=source_classes[j],
                                       split='train', transform=train_transform)

        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
        train_source_iter = ForeverDataIterator(train_source_loader)
        train_source_iter_list.append(train_source_iter)

    val_loader = []
    for j, the_source in enumerate(args.source):
        val_source_dataset = dataset(root=args.root, task=the_source, filter_class=source_classes[j],
                                       split='val', transform=val_tranform)

        val_source_loader = DataLoader(val_source_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)

        val_loader.append(val_source_loader)

    test_dataset = dataset(root=args.root, task=args.target, filter_class=ST1,
                                   split='all', transform=val_tranform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)
    test_loader = [test_loader]

    # create model
    num_classes = n1 + 3 * n2 + 3 * n3

    backbone1 = resnet18_fast()
    backbone2 = resnet18_fast()
    backbone3 = resnet18_fast()

    classifier = ClassifierFast(backbone1, backbone2, backbone3, num_classes).cuda()

    # define optimizer and lr scheduler
    if args.data == 'PACS':
        nesterov = False
        weight_decay = 0.0001
    elif args.data == 'OfficeHome':
        nesterov = True
        weight_decay = args.wd

    optimizer = SGD(classifier.get_parameters(), args.lr,
                    momentum=args.momentum, weight_decay=weight_decay, nesterov=nesterov)

    step_size = int(args.epochs * .8)
    lr_sheduler = StepLR(optimizer, init_lr=args.lr, gamma=0.1, step_size=step_size)


    # start training
    best_val_acc1 = 0.
    best_test_acc1 = 0.

    for epoch in range(args.epochs):
        print(lr_sheduler.get_lr())
        lr_sheduler.step()

        # train for one epoch

        train(train_source_iter_list, classifier, optimizer,
              lr_sheduler, epoch, args)

        # evaluate on validation set
        print('Validation Set')
        val_acc1 = validate(val_loader, classifier, args)
        print('Test Set')
        test_acc1 = validate(test_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        if val_acc1 > best_val_acc1:
            best_val_model = copy.deepcopy(classifier.state_dict())
            torch.save(best_val_model, val_save_dir)
            print('ValCurrent best acc')
            best_val_acc1 = max(val_acc1, best_val_acc1)

        # remember best acc@1 and save checkpoint
        if test_acc1 > best_test_acc1:
            best_test_model = copy.deepcopy(classifier.state_dict())
            torch.save(best_test_model, test_save_dir)
            print('TestCurrent best acc')
            best_test_acc1 = max(test_acc1, best_test_acc1)

    torch.save(classifier.state_dict(), final_save_dir)

    classifier.load_state_dict(best_val_model)
    best_val_test_acc1 = validate(test_loader, classifier, args)
    print("best_val_acc1 = {:3.1f}, {:3.1f}".format(best_val_acc1, best_val_test_acc1))
    print("best_test_acc1 = {:3.1f}".format(best_test_acc1))


def create_one_hot(y, classes):
    y_onehot = torch.LongTensor(y.size(0), classes).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot

def get_sample_mixup_random(domains):
    indeces = torch.randperm(domains.size(0))
    return indeces.long()


def get_ratio_mixup_Dirichlet(domains, mixup_dir_list):
    return torch.from_numpy(RG.dirichlet(mixup_dir_list, size=domains.size(0))).float()   # N * 3


def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
    return loss


def DistillKL(y_s, y_t, T):
    """KL divergence for distillation"""
    p_s = F.log_softmax(y_s/T, dim=1)
    p_t = y_t
    loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]
    return loss


def train(train_source_iter_list, model, optimizer,
          lr_sheduler, epoch, args):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    cls_losses = AverageMeter('C Loss', ':3.2f')
    kd_losses = AverageMeter('KD Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    mixup_losses = AverageMeter('Dir Loss', ':3.2f')
    dirichlet_losses = AverageMeter('ValDir Loss', ':3.2f')

    val_cls_losses = AverageMeter('ValC Loss', ':3.2f')
    val_cls_accs = AverageMeter('Val Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs, kd_losses, mixup_losses,
         val_cls_losses, val_cls_accs, dirichlet_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):

        # measure data loading time
        data_time.update(time.time() - end)

        # -----------------  Meta Train  -----------------------------------
        meta_train_loss = 0.0
        fast_parameters = list(model.parameters())
        for weight in model.parameters():
            weight.fast = None
        model.zero_grad()

        total_all_f_s = [[], [], []]   # model_domains * 3batch_size
        all_one_hot_labels = []  # 3batch_size

        for data_domain, train_source_iter in enumerate(train_source_iter_list):
            x_s, labels_s, _ = next(train_source_iter)
            x_s = x_s.cuda()
            labels_s = labels_s.cuda()
            one_hot_labels = create_one_hot(labels_s, model.num_classes)
            all_one_hot_labels.append(one_hot_labels)

            # compute output
            y_s_distill = []
            for model_domain in range(3):
                y_s, f_s = model(x_s, domain=model_domain)
                if model_domain != data_domain:
                    y_s_distill.append(y_s)
                else:
                    y_s_pred = y_s
                total_all_f_s[model_domain].append(f_s)

            cls_loss = F.cross_entropy(y_s_pred, labels_s)
            meta_train_loss = meta_train_loss + cls_loss

            # Distill
            y_s_distill = torch.stack(y_s_distill) # 2 * N * C
            y_s_distill = F.softmax(y_s_distill/args.T, dim=2)
            domains = [0] * args.batch_size
            domains = torch.LongTensor(domains)
            mixup_ratios = get_ratio_mixup_Dirichlet(domains, [1.0, 1.0])
            mixup_ratios = mixup_ratios.cuda()  # N * 2
            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1) # 2 * N * 1
            y_s_distill = torch.sum(y_s_distill * mixup_ratios, dim=0)
            kd_loss = DistillKL(y_s_pred, y_s_distill.detach(), args.T)
            meta_train_loss = meta_train_loss + args.trade2 * kd_loss
            kd_losses.update(kd_loss.item(), x_s.size(0))

            cls_acc = accuracy(y_s_pred, labels_s)[0]
            cls_accs.update(cls_acc.item(), x_s.size(0))
            cls_losses.update(cls_loss.item(), x_s.size(0))

        # Dirichlet Mixup
        all_one_hot_labels = torch.cat(all_one_hot_labels, dim=0)
        mixup_loss = 0.0

        for model_domain in range(3):
            # MixUp
            all_f_s = torch.cat(total_all_f_s[model_domain], dim=0)
            domains = [0] * args.batch_size
            domains = torch.LongTensor(domains)

            all_f_s_1 = all_f_s[(0 * args.batch_size):((0 + 1) * args.batch_size)]
            all_f_s_2 = all_f_s[(1 * args.batch_size):((1 + 1) * args.batch_size)]
            all_f_s_3 = all_f_s[(2 * args.batch_size):((2 + 1) * args.batch_size)]

            all_one_hot_labels_1 = all_one_hot_labels[(0 * args.batch_size):((0 + 1) * args.batch_size)]
            all_one_hot_labels_2 = all_one_hot_labels[(1 * args.batch_size):((1 + 1) * args.batch_size)]
            all_one_hot_labels_3 = all_one_hot_labels[(2 * args.batch_size):((2 + 1) * args.batch_size)]

            mixup_dir_list = [args.mixup_dir2, args.mixup_dir2, args.mixup_dir2]
            mixup_dir_list[model_domain] = args.mixup_dir

            mixup_ratios = get_ratio_mixup_Dirichlet(domains, mixup_dir_list)
            mixup_ratios = mixup_ratios.cuda()  # N * 3
            mix_indeces_1 = get_sample_mixup_random(domains)
            mix_indeces_2 = get_sample_mixup_random(domains)
            mix_indeces_3 = get_sample_mixup_random(domains)

            mixup_features = torch.stack(
                [all_f_s_1[mix_indeces_1], all_f_s_2[mix_indeces_2], all_f_s_3[mix_indeces_3]])  # 3 * N * D
            mixup_labels = torch.stack([all_one_hot_labels_1[mix_indeces_1],
                                        all_one_hot_labels_2[mix_indeces_2],
                                        all_one_hot_labels_3[mix_indeces_3]])  # 3 * N * C

            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)

            mixup_features = torch.sum((mixup_features * mixup_ratios), dim=0)
            mixup_labels = torch.sum((mixup_labels * mixup_ratios), dim=0)

            mixup_features_predictions = model.heads[model_domain](mixup_features)
            mixup_feature_loss = manual_CE(mixup_features_predictions, mixup_labels)

            mixup_loss = mixup_loss + mixup_feature_loss
            mixup_losses.update(mixup_feature_loss.item(), all_f_s.size(0) / 3)

        meta_train_loss = meta_train_loss + args.trade * mixup_loss

        # -----------------  Meta Objective -----------------------------------
        meta_val_loss = 0.0

        grad = torch.autograd.grad(meta_train_loss, fast_parameters,
                                   create_graph=True)

        if args.stop_gradient:
            grad = [g.detach() for g in
                    grad]

        fast_parameters = []
        for k, weight in enumerate(model.parameters()):
            if weight.fast is None:
                weight.fast = weight - args.meta_step_size * grad[k]
            else:
                weight.fast = weight.fast - args.meta_step_size * grad[
                    k]
            fast_parameters.append(
                weight.fast)

        total_all_f_s = [[], [], []]  # model_domains * 3batch_size
        all_one_hot_labels = []  # 3batch_size

        for data_domain, train_source_iter in enumerate(train_source_iter_list):
            x_s, labels_s, _ = next(train_source_iter)
            x_s = x_s.cuda()
            labels_s = labels_s.cuda()
            one_hot_labels = create_one_hot(labels_s, model.num_classes)
            all_one_hot_labels.append(one_hot_labels)

            # compute output
            y_s_list = []
            for model_domain in range(3):
                y_s, f_s = model(x_s, domain=model_domain)
                y_s_list.append(y_s)
                total_all_f_s[model_domain].append(f_s)

                if model_domain != data_domain:
                    cls_loss = F.cross_entropy(y_s, labels_s)
                    meta_val_loss = meta_val_loss + args.trade3 * cls_loss

                    cls_acc = accuracy(y_s, labels_s)[0]
                    val_cls_accs.update(cls_acc.item(), x_s.size(0))
                    val_cls_losses.update(cls_loss.item(), x_s.size(0))

        all_one_hot_labels = torch.cat(all_one_hot_labels, dim=0)

        # Dirichelet Mixup
        mixup_loss_dirichlet = 0.0

        for model_domain in range(3):
            # MixUp
            all_f_s = torch.cat(total_all_f_s[model_domain], dim=0)
            domains = [0] * args.batch_size
            domains = torch.LongTensor(domains)

            all_f_s_1 = all_f_s[(0*args.batch_size):((0+1)*args.batch_size)]
            all_f_s_2 = all_f_s[(1*args.batch_size):((1+1)*args.batch_size)]
            all_f_s_3 = all_f_s[(2*args.batch_size):((2+1)*args.batch_size)]

            all_one_hot_labels_1 = all_one_hot_labels[(0*args.batch_size):((0+1)*args.batch_size)]
            all_one_hot_labels_2 = all_one_hot_labels[(1*args.batch_size):((1+1)*args.batch_size)]
            all_one_hot_labels_3 = all_one_hot_labels[(2*args.batch_size):((2+1)*args.batch_size)]

            mixup_dir_list = [args.mixup_dir, args.mixup_dir, args.mixup_dir]
            mixup_dir_list[model_domain] = args.mixup_dir2

            mixup_ratios = get_ratio_mixup_Dirichlet(domains, mixup_dir_list)
            mixup_ratios = mixup_ratios.cuda()  # N * 3
            mix_indeces_1 = get_sample_mixup_random(domains)
            mix_indeces_2 = get_sample_mixup_random(domains)
            mix_indeces_3 = get_sample_mixup_random(domains)

            mixup_features = torch.stack([all_f_s_1[mix_indeces_1], all_f_s_2[mix_indeces_2], all_f_s_3[mix_indeces_3]])    # 3 * N * D
            mixup_labels = torch.stack([all_one_hot_labels_1[mix_indeces_1],
                                        all_one_hot_labels_2[mix_indeces_2], all_one_hot_labels_3[mix_indeces_3]])    # 3 * N * C

            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)

            mixup_features = torch.sum((mixup_features * mixup_ratios), dim=0)
            mixup_labels = torch.sum((mixup_labels * mixup_ratios), dim=0)

            mixup_features_predictions = model.heads[model_domain](mixup_features)
            mixup_feature_loss_dirichlet = manual_CE(mixup_features_predictions, mixup_labels)

            mixup_loss_dirichlet = mixup_loss_dirichlet + mixup_feature_loss_dirichlet
            dirichlet_losses.update(mixup_feature_loss_dirichlet.item(), all_f_s.size(0) / 3)

        meta_val_loss = meta_val_loss + args.trade4 * mixup_loss_dirichlet

        total_loss = meta_train_loss + meta_val_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loaders, model, args):

    all_acc = []

    for weight in model.parameters():
        weight.fast = None

    for val_loader in val_loaders:
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')

        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target, _) in enumerate(val_loader):
                images = images.cuda()
                target = target.cuda()

                # compute output
                [output_0, output_1, output_2], _ = model(images, -1)

                output = F.softmax(output_0, 1) + F.softmax(output_1, 1) + F.softmax(output_2, 1)
                output = output / 3

                # measure accuracy and record loss
                acc1, _ = accuracy(output, target, topk=(1, 5))

                top1.update(acc1.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            print(' * Acc@1 {top1.avg:.3f}'
                  .format(top1=top1))

        all_acc.append(top1.avg)

    all_acc_ = np.mean(all_acc)
    print('Mean validation acc %.3f.' % all_acc_)
    return all_acc_


if __name__ == '__main__':

    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Open Domain Generalization')
    parser.add_argument('--root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='OfficeHome',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: OfficeHome)')
    parser.add_argument('-s', '--source', type=str, help='source domain(s)')
    parser.add_argument('-t', '--target', type=str, help='target domain(s)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        metavar='N',
                        help='mini-batch size (default: 24)')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-i', '--iters_per_epoch', default=101, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id ')
    parser.add_argument('--savename', default='saved', type=str, help='saved name of the model')
    parser.add_argument("--trade", type=float, default=3.0)
    parser.add_argument("--trade2", type=float, default=1.0)
    parser.add_argument("--trade3", type=float, default=1.0)
    parser.add_argument("--trade4", type=float, default=3.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--mixup_dir", type=float, default=0.6)
    parser.add_argument("--mixup_dir2", type=float, default=0.2)
    parser.add_argument("--stop_gradient", type=int, default=1,
                        help='whether stop gradient of the first order gradient')
    parser.add_argument("--meta_step_size", type=float, default=0.01)

    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

