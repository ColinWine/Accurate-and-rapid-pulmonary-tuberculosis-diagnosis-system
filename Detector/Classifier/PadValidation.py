from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from RandomSelectionData import SliceData
import se_resnet
import matplotlib.pyplot as plt
import numpy as np
import itertools

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
	# 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
	# x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
	# 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def train_model(args, model, criterion, dataloders, dataset_sizes):
    since = time.time()

    # Each epoch has a training and validation phase
    model.train(False)  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    img_cnt = 0

    conf_matrix = torch.zeros(6, 6)
    for i, (inputs, labels) in enumerate(dataloders['val']):
        # wrap them in Variable
        for img_num in range(len(labels)):
            plt.imshow(inputs[img_num][0])
            #print(inputs[img_num][0].max())
            save_name = os.path.join(r'L:\Classification\visual',str(img_cnt+img_num))
            #plt.savefig()
            plt.savefig(save_name)
            plt.close('all')

        img_cnt = img_cnt+len(labels)

        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        # forward
        print(inputs.shape)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        conf_matrix = confusion_matrix(preds, labels=labels.data, conf_matrix=conf_matrix)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == labels.data)



        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects.cpu().numpy() / dataset_sizes['val']
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


    time_elapsed = time.time() - since
    plot_confusion_matrix(conf_matrix.numpy(), classes=['Calcified granulomas','Cavitation','Centrilobular and tree-in-bud nodules','Clusters of nodules','Consolidation','Fibronodular scarring'], normalize=False,title='Normalized confusion matrix')
    print('Validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=6)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output2")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet_18", help="")
    args = parser.parse_args()

    # read data
    dataloders, dataset_sizes = SliceData(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])

    if script_name == "se_resnet":
        model = getattr(se_resnet ,args.network)(num_classes = args.num_class)
    else:
        raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            #base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        torch.backends.cudnn.enabled = False
        #model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.CrossEntropyLoss()


    model = train_model(args=args,model=model,criterion=criterion,dataloders=dataloders, dataset_sizes=dataset_sizes)
