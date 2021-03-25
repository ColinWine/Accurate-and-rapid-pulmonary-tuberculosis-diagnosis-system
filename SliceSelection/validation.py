from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy
import time
import os
import argparse
from SliceDataLoader import PatientData

import sse_resnet

def train_model(args, model, criterion, patient_list, dataloders, dataset_sizes):
    since = time.time()
    model.train(False)  # Set model to evaluate mode
    # Iterate over data.
    print('start validation')
    count = 0
    all_acc = 0
    for patient in patient_list:
        print('Patient: ',patient)
        dataloder = dataloders[patient]
        running_corrects = 0
        prob_list = []
        pred_list = []
        correct_list = []
        for i, (inputs, labels) in enumerate(dataloder):
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)


            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            for prob in outputs.data.cpu().numpy():
                prob_list.append(prob[1])

            for pred in preds.cpu().numpy():
                pred_list.append(pred)

            for correct in (preds == labels.data).cpu().numpy():
                correct_list.append(correct)
            # statistics
            running_corrects += torch.sum(preds == labels.data)

        prob_array = numpy.array(prob_list)
        correct_array = numpy.array(correct_list)
        top_10_index = prob_array.argsort()[-10:][::-1]
        print(correct_array[top_10_index])
        print(prob_array[top_10_index])
        count = count + 1
        all_acc = all_acc + correct_array[top_10_index].mean()
        print('overall acc:', all_acc/count)

    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output2")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="SSe_resnet_18", help="")
    args = parser.parse_args()

    # read data
    patient_list, dataloders, dataset_sizes = PatientData(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])
    model = getattr(sse_resnet ,args.network)(num_classes = args.num_class)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        torch.backends.cudnn.enabled = False

    # define loss function
    criterion = nn.CrossEntropyLoss()

    model = train_model(args=args,model=model,criterion=criterion,patient_list=patient_list,dataloders=dataloders, dataset_sizes=dataset_sizes)
