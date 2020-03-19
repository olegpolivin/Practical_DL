import argparse
import numpy as np
import pandas as pd
import os
import skimage
from skimage import io
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint_sequential
import time


class TestDataset(Dataset):
    def __init__(self, root_folder, labels_frame, class_to_idx, transform=None):

        self.transform = transform
        self.root_folder = root_folder
        self.labels_frame = labels_frame
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels_frame) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_folder,
                                self.labels_frame.loc[idx, 'imname'])
        image = io.imread(img_name)

        # Treating greyscale images
        if len(image.shape) < 3:
            image = skimage.color.grey2rgb(image, alpha=None)

        if self.transform:
            image = self.transform(image)
        category = self.class_to_idx[self.labels_frame.loc[idx, 'id']]
        return image, category


class MyGreatModel(nn.Module):

    def __init__(self):
        super(MyGreatModel, self).__init__()

        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv2d(3, 200, kernel_size=(3, 3), stride=1))
        self.model.add_module('bn1_1', nn.BatchNorm2d(200))
        self.model.add_module('relu1_1', nn.ReLU())
        self.model.add_module('conv1_2', nn.Conv2d(200, 200, kernel_size=(3, 3), stride=1))
        self.model.add_module('bn1_2', nn.BatchNorm2d(200))
        self.model.add_module('relu1_2', nn.ReLU())
        self.model.add_module('maxpool1', nn.MaxPool2d(3))

        self.model.add_module('conv2_1', nn.Conv2d(200, 400, kernel_size=(3, 3), stride=1))
        self.model.add_module('bn2_1', nn.BatchNorm2d(400))
        self.model.add_module('relu2_1', nn.ReLU())
        self.model.add_module('conv2_2', nn.Conv2d(400, 400, kernel_size=(3, 3), stride=1))
        self.model.add_module('bn2_2', nn.BatchNorm2d(400))
        self.model.add_module('relu2_2', nn.ReLU())
        self.model.add_module('maxpool2', nn.MaxPool2d(3))

        self.model.add_module('flatten', nn.Flatten())
        self.model.add_module('fc1', nn.Linear(10000, 1000))
        self.model.add_module('dp1', nn.Dropout(0.5))
        self.model.add_module('fc2', nn.Linear(1000, 200))

    def forward(self, x):
        x = self.model(x)
        return x


class CheckpointedModel(nn.Module):

    def __init__(self, arrangement_num=0):
        super(CheckpointedModel, self).__init__()
        self.arrangement_num = arrangement_num

        self.features_1 = nn.Sequential(
            nn.Conv2d(3, 200, kernel_size=(3, 3), stride=1)
        )
        self.bn_1 = nn.BatchNorm2d(200)
        self.features_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=(3, 3), stride=1)
        )
        self.bn_2 = nn.BatchNorm2d(200)
        self.features_3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(200, 400, kernel_size=(3, 3), stride=1)
        )
        self.bn_3 = nn.BatchNorm2d(400)
        self.features_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(400, 400, kernel_size=(3, 3), stride=1)
        )
        self.bn_4 = nn.BatchNorm2d(400)
        self.features_5 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(10000, 1000)
        )
        self.dp1 = nn.Dropout(0.5)
        self.classifier = nn.Linear(1000, 200)

    def forward(self, x):
        '''
        Arrangement 0: as it is, only blocks, no checkpoiting
        Arrangement 1: Added checkpoint_sequential(self.features_1, 1, x)
        etc...
        '''
        arrangement = self.arrangement_num

        # Arrangement 0
        if arrangement == 0:
            x = self.features_1(x)
            x = self.bn_1(x)
            x = self.features_2(x)
            x = self.bn_2(x)
            x = self.features_3(x)
            x = self.bn_3(x)
            x = self.features_4(x)
            x = self.bn_4(x)
            x = self.features_5(x)
            x = self.dp1(x)
            x = self.classifier(x)
            return x

        # Arrangement 1
        if arrangement == 1:
            x = checkpoint_sequential(self.features_1, 1, x)
            x = self.bn_1(x)
            x = self.features_2(x)
            x = self.bn_2(x)
            x = self.features_3(x)
            x = self.bn_3(x)
            x = self.features_4(x)
            x = self.bn_4(x)
            x = self.features_5(x)
            x = self.dp1(x)
            x = self.classifier(x)
            return x

        # Arrangement 2
        if arrangement == 2:
            x = checkpoint_sequential(self.features_1, 1, x)
            x = self.bn_1(x)
            x = checkpoint_sequential(self.features_2, 2, x)
            x = self.bn_2(x)
            x = self.features_3(x)
            x = self.bn_3(x)
            x = self.features_4(x)
            x = self.bn_4(x)
            x = self.features_5(x)
            x = self.dp1(x)
            x = self.classifier(x)
            return x

        # Arrangement 3
        if arrangement == 3:
            x = checkpoint_sequential(self.features_1, 1, x)
            x = self.bn_1(x)
            x = checkpoint_sequential(self.features_2, 2, x)
            x = self.bn_2(x)
            x = checkpoint_sequential(self.features_3, 3, x)
            x = self.bn_3(x)
            x = self.features_4(x)
            x = self.bn_4(x)
            x = self.features_5(x)
            x = self.dp1(x)
            x = self.classifier(x)
            return x

        # Arrangement 4
        if arrangement == 4:
            x = checkpoint_sequential(self.features_1, 1, x)
            x = self.bn_1(x)
            x = checkpoint_sequential(self.features_2, 2, x)
            x = self.bn_2(x)
            x = checkpoint_sequential(self.features_3, 3, x)
            x = self.bn_3(x)
            x = checkpoint_sequential(self.features_4, 2, x)
            x = self.bn_4(x)
            x = self.features_5(x)
            x = self.dp1(x)
            x = self.classifier(x)
            return x

        # Arrangement 5
        if arrangement == 5:
            x = checkpoint_sequential(self.features_1, 1, x)
            x = self.bn_1(x)
            x = checkpoint_sequential(self.features_2, 2, x)
            x = self.bn_2(x)
            x = checkpoint_sequential(self.features_3, 3, x)
            x = self.bn_3(x)
            x = checkpoint_sequential(self.features_4, 2, x)
            x = self.bn_4(x)
            x = checkpoint_sequential(self.features_5, 4, x)
            x = self.dp1(x)
            x = self.classifier(x)
            return x


def load_train_data(batch_size):
    """Function to load data, split to train/validation set.
    A predefined list of transforms is applied.
    """

    transform_train = transforms.Compose([
        transforms.RandomChoice(
            [transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomAffine(2, scale=(1, 1.3))]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train',
                                                        transform=transform_train)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])

    # Dictionary that maps classes to indices
    class_to_idx = dataset.class_to_idx
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
    return train_loader, val_loader, class_to_idx


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    loss_fn,
    batches_per_update,
    scheduler=None):

    train_loss = []
    val_accuracy = []
    ave_time_per_batch = 0
    model.train(True)

    for epoch in range(num_epochs):
        start_time = time.time()
        start_epoch_counter = time.perf_counter()

        # 1 Clear the gradients
        optimizer.zero_grad()

        for batch_i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # I. Training

            # 2 Forward
            start_counter = time.perf_counter()
            predictions = model.forward(x_batch)
            elapsed_time = time.perf_counter() - start_counter
            ave_time_per_batch += elapsed_time/batch_size

            # 3 Calculating loss
            loss = loss_fn(predictions, y_batch)

            # 4 Calculating gradients
            loss.backward()

            # 5 Optimizer step - Adapted for accumulation of gradients
            if (batch_i + 1) % batches_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()

            # II. Tracking the training
            train_loss.append(loss.cpu().data.numpy())
        if scheduler:
            scheduler.step()

        elapsed_epoch_time = time.perf_counter() - start_epoch_counter

        # III. Validation
        model.train(False)  # disable dropout / use averages for batch_norm
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model.forward(x_batch)
            y_pred = predictions.max(1)[1].data
            val_accuracy.append(np.mean( (y_batch.cpu() == y_pred.cpu()).numpy() ))

        validation_accuracy = np.mean(val_accuracy[-20000 // batch_size:]) * 100
        # IV. Reporting
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  (added) training part takes {:.3f}s to complete".format(elapsed_epoch_time))
        print("  (added) on average one batch of size {}s is processed in {:.3f}s".format(
            batch_size, ave_time_per_batch))
        print("  training loss (in-iteration): \t{:.6f}".format(
            np.mean(train_loss[-80000 // batch_size:])))
        print("  validation accuracy: \t\t\t{:.2f} %".format(
            validation_accuracy))

        if validation_accuracy > 40:
            print(f'Fitted the model to exceed 40% on the validation set. Exiting loop on epoch {epoch + 1}.')
            break
    return model, train_loss, val_accuracy


def evaluate_on_test_data(batch_size, class_to_idx):

    print('Evaluating on test data')
    labels = pd.read_csv('tiny-imagenet-200/val/val_annotations.txt', sep='\t', header=None)
    labels.columns = ['imname', 'id', 'bb1', 'bb2', 'bb3', 'bb4']

    test_dataset = TestDataset(root_folder='tiny-imagenet-200/val/images/',
                           labels_frame=labels,
                           class_to_idx=class_to_idx,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                           ]))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)

    model.train(False)
    correct_samples = 0
    total_samples = 0
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        predictions = model.forward(x_batch)
        y_pred = predictions.max(1)[1].data
        correct_samples += torch.sum(y_pred == y_batch)
        total_samples += y_batch.shape[0]
    test_accuracy = float(correct_samples) / total_samples
    print("Final results:")
    print("  test accuracy:\t\t{:.2f} %".format(
        test_accuracy * 100))

    if test_accuracy * 100 > 40:
        print("Achievement unlocked: 110lvl Warlock!")
    elif test_accuracy * 100 > 35:
        print("Achievement unlocked: 80lvl Warlock!")
    elif test_accuracy * 100 > 30:
        print("Achievement unlocked: 70lvl Warlock!")
    elif test_accuracy * 100 > 25:
        print("Achievement unlocked: 60lvl Warlock!")
    else:
        print("We need more magic! Follow instructons below")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a convolutional neural network on the Tiny-Imagenet dataset.')
    parser.add_argument('--bs', type=int, default=64, help='batch size that will be used for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='num epochs to train')
    parser.add_argument('--wd', type=float, default=0.00001, help='regularization coef')
    parser.add_argument('--gamma', type=float, default=0.1, help='decrease lr by this coef every 8 epochs')
    parser.add_argument('--evaluate', action='store_true', help='evaluate on the test set')
    parser.add_argument('--sched', action='store_true', help='use scheduler or not')
    parser.add_argument('--arrangement_num', type=int, default=None, help='use checkpointing arrangement')
    parser.add_argument('--ebs', type=int, default=None, help='effective batch size for gradient accumulation')

    args = parser.parse_args()

    batch_size = args.bs
    lr = args.lr
    num_epochs = args.num_epochs
    weight_decay = args.wd
    gamma = args.gamma
    evaluate = args.evaluate
    use_scheduler = args.sched
    arrangement_num = args.arrangement_num
    effective_batch_size = args.ebs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_to_idx = load_train_data(batch_size)

    if arrangement_num:
        model = CheckpointedModel(arrangement_num)
    else:
        model = MyGreatModel()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # L2 regularization is added through weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=gamma)

    if effective_batch_size:
        batches_per_update = effective_batch_size / batch_size
    else:
        batches_per_update = 1  # That is update every batch

    model, train_loss, val_accuracy = train_model(
                                                model,
                                                train_loader,
                                                val_loader,
                                                optimizer,
                                                num_epochs,
                                                device,
                                                loss_fn,
                                                batches_per_update,
                                                scheduler=scheduler)
    if evaluate:
        evaluate_on_test_data(batch_size, class_to_idx)

    # Evaluate memory usage
    print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")
