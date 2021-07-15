import sys
import argparse


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')

import dalib.vision.datasets as datasets
from tools.utils import AverageMeter, accuracy

import os
import numpy as np

from tools.daml_utils import resnet18_fast, ClassifierFast


def main(args):
    """
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    """
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_tranform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if args.data == 'PACS':
        n1 = 0
        n2 = 1
        n3 = 1
        n4 = 0
        n5 = 1
        n6 = 1
        n7 = 1
    elif args.data == 'OfficeHome':
        n1 = 3
        n2 = 6
        n3 = 11
        n4 = 1
        n5 = 2
        n6 = 3
        n7 = 11

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

    TT = [i + n1 + n2 * 3 + n3 * 3 for i in range(n7)]

    source_classes = [[], [], []]
    source_classes[0] = S1 + S12 + S13 + S123
    source_classes[1] = S2 + S12 + S23 + S123
    source_classes[2] = S3 + S13 + S23 + S123

    all_target_classes = ST1 + TT
    print(all_target_classes)
    
    dataset = datasets.__dict__[args.data]

    num_classes = n1 + n2 * 3 + n3 * 3

    backbone1 = resnet18_fast()
    backbone2 = resnet18_fast()
    backbone3 = resnet18_fast()

    classifier = ClassifierFast(backbone1, backbone2, backbone3, num_classes).cuda()
    for weight in classifier.parameters():
        weight.fast = None

    pretrained_dir = 'runs/daml-' + args.source + '-' + args.target + '_best_val.tar'

    classifier.load_state_dict(torch.load(pretrained_dir))
    the_target = args.target
    print('Target Domain: ', the_target)

    raw_output_dict = {}
    target_dict = {}
    targets = []

    for class_index in all_target_classes:
        the_class = [class_index]

        the_target_dataset = dataset(root=args.root, task=the_target, filter_class=the_class, split='all',
                                    transform=val_tranform)
        the_target_loader = DataLoader(the_target_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.workers, drop_last=False)
        the_target_list = [the_target_loader]

        output, target = get_raw_output(the_target_list, classifier, num_classes)
        raw_output_dict[class_index] = output
        target_dict[class_index] = target
        targets.append(target)

    targets = torch.cat(targets)
    outlier_targets = (targets == num_classes).float()

    T = 1.0
    tsm_output_dict = {}
    outlier_indi_dict = {}
    outlier_indis = []

    for class_index in all_target_classes:
        raw_output = raw_output_dict[class_index]

        output, indicator = get_new_output(raw_output, T)

        tsm_output_dict[class_index] = output
        outlier_indi_dict[class_index] = indicator
        outlier_indis.append(indicator)

    outlier_indis = torch.cat(outlier_indis)

    thd_min = torch.min(outlier_indis)
    thd_max = torch.max(outlier_indis)
    outlier_range = [thd_min + (thd_max - thd_min) * i / 9 for i in range(10)]

    best_overall_acc = 0.0
    best_thred_acc = 0.0
    best_overall_Hscore = 0.0
    best_thred_Hscore = 0.0
    best_overall_caa = 0.0
    best_thred_caa = 0.0

    for outlier_thred in outlier_range:
        acc_dict = {}
        for class_index in all_target_classes:
            tsm_output = tsm_output_dict[class_index]
            outlier_indi = outlier_indi_dict[class_index]
            target = target_dict[class_index]
            acc = get_acc(tsm_output, outlier_indi, outlier_thred, target)
            acc_dict[class_index] = acc

        overall_acc = (np.sum([acc.sum.item() for acc in acc_dict.values()]) / np.sum([acc.count for acc in acc_dict.values()])).item()

        insider = (np.sum([acc_dict[Cl].sum.item() for Cl in ST1]) / np.sum([acc_dict[Cl].count for Cl in ST1])).item()
        outsider = (np.sum([acc_dict[Cl].sum.item() for Cl in TT]) / np.sum([acc_dict[Cl].count for Cl in TT])).item()
        overall_Hscore = 2.0 * insider * outsider / (insider + outsider)

        overall_caa = np.mean([acc.avg.item() for acc in acc_dict.values()])

        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            best_thred_acc = outlier_thred
        if overall_Hscore > best_overall_Hscore:
            best_overall_Hscore = overall_Hscore
            best_thred_Hscore = outlier_thred
        if overall_caa > best_overall_caa:
            best_overall_caa = overall_caa
            best_thred_caa = outlier_thred


    print('Best OverallAcc: %.2f' % (best_overall_acc), 'Best threshold Acc: %.3f' % (best_thred_acc),
    'Best OverallHscore: %.2f' % (best_overall_Hscore), 'Best threshold Hscore: %.3f' % (best_thred_Hscore),
    'Best OverallCaa: %.2f' % (best_overall_caa), 'Best threshold Caa: %.3f' % (best_thred_caa))


def get_raw_output(val_loader, model, num_classes):
    model.eval()
    output_sum = []
    target_sum = []

    with torch.no_grad():
        for the_loader in val_loader:
            for i, (images, target, _) in enumerate(the_loader):
                images = images.cuda()
                target = target.cuda()
                outlier_flag = (target > (num_classes - 1)).float()
                target = target * (1 - outlier_flag) + num_classes * outlier_flag
                target = target.long()

                output, _ = model(images)
                output_sum.append(output)

                target_sum.append(target)

    output_sum = [torch.cat([output_sum[j][i] for j in range(len(output_sum))], dim=0) for i in range(3)]

    target_sum = torch.cat(target_sum)
    return output_sum, target_sum


def get_new_output(raw_output, T):

    output = [F.softmax(headout/T, dim=1) for headout in raw_output]
    output = torch.mean(torch.stack(output), 0)
    max_prob, max_index = torch.max(output, 1)
    return output, max_prob



def get_acc(tsm_output, outlier_indi, outlier_thred, target):
    top1 = AverageMeter('Acc@1', ':6.2f')

    outlier_pred = (outlier_indi < outlier_thred).float()
    outlier_pred = outlier_pred.view(-1, 1)
    output = torch.cat((tsm_output, outlier_pred.cuda()), dim=1)
    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    top1.update(acc1[0], output.shape[0])
    return top1



if __name__ == '__main__':

    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Open Domain Generalization')
    parser.add_argument('--root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', type=str, help='source domain(s)')
    parser.add_argument('-t', '--target', type=str, help='target domain(s)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id ')

    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

