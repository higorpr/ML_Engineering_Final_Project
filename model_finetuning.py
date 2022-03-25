# Import Packages

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import ImageFile

def test(model, test_loader, device, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    model.eval() # Set model to evaluation mode
    running_loss = 0 # Sum of losses for average loss calculation
    running_corrects = 0 # Sum of corrects predictions for accuracy calculation

    for input_tensor, labels in test_loader: # Iteration over data
        input_tensor = input_tensor.to(device) # Send data being evaluated to chosen device
        labels = labels.to(device) # Send label being evaluated to chosen device

        # Get prediction for each input:
        output = model(input_tensor)
        loss = criterion(output,labels)
        percentage, preds = torch.max(output, 1)
        running_loss += loss.item() * input_tensor.size(0)
        running_corrects += torch.sum(preds == labels).item()

    # End of iteration over test loader data

    # Calculation of average loss
    avg_loss = running_loss / len(test_loader.dataset)

    # Calculation of accuracy
    acc = running_corrects / len(test_loader.dataset)

    print('Printing Log')
    print(
        "\nTest set average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            avg_loss, running_corrects, len(test_loader.dataset), 100.0 * acc
        )
    )

def train(model, epochs, train_loader, evaluation_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    loader = {'train': train_loader, 'evaluation': evaluation_loader} # Creation of loader to be used in iteration
    epochs = epochs # Defining number of epochs for training
    best_loss = 1e6
    loss_counter = 0

    for epoch in range(epochs): # Going over all epochs
        for phase in ['train', 'evaluation']: # Variation for training and evaluation phases
            print(f'Epoch: {epoch} / Phase: {phase}')

            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Initialize cumulative variables for each epoch:
            accum_loss = 0
            accum_corrects = 0
            samples_ran = 0

            for input_tensor, labels in loader[phase]: # Iteration over data in loader
                input_tensor = input_tensor.to(device)
                labels = labels.to(device)

                # Getting output tensor:
                output = model(input_tensor)
                # Getting loss value (as tensor)
                loss = criterion(output, labels)

                # Section that is different between train and evaluation:
                if phase == 'train':
                    optimizer.zero_grad() # Zeroes the gradients in the optimizer
                    loss.backward() # Performs backward pass
                    optimizer.step() # Updates parameters of Neural Network with the calculated gradients

                percentages, preds = torch.max(output, dim=1)
                accum_loss += loss.item() * input_tensor.size(0)
                accum_corrects += torch.sum(preds == labels).item()
                samples_ran += len(input_tensor)

                # Printing Log after a number of predictions (samples ran):
                if samples_ran % 2000 == 0:
                    accuracy = accum_corrects / samples_ran
                    print(
                        'Images [{}/{} ({:.0f}%)] / Loss: {:.2f} / Accumulated Epoch Loss: {:.0f} / '
                        'Accuracy: {}/{} ({:.2f}%)'.format(
                            samples_ran, # Number of images analyzed
                            len(loader[phase].dataset), # Total images to be analyzed in the loader
                            100 * (samples_ran / len(loader[phase].dataset)), # Percentage of images already analyzed
                            loss.item(), # Loss for this prediction
                            accum_loss, # Accumulated loss for this epoch
                            accum_corrects, # Number of correct predictions for this epoch
                            samples_ran,
                            100 * accuracy
                        ))

                # Section that train and evaluates on a portion of the dataset (Used to test the train script)
                # Comment to train on the whole dataset:b
#                 if samples_ran >= (0.2 * len(loader[phase].dataset)):
#                     break # Stops the epoch

                # End of training and evaluation phases for an epoch

            # Convergence Check (done after each evaluation phase)
            if phase == 'evaluation':
                avg_epoch_loss = accum_loss / samples_ran
                if avg_epoch_loss < best_loss: # Loss still reducing (converging)
                    best_loss = avg_epoch_loss
                else: # Loss increased (diverging)
                    loss_counter += 1

        # If the loss is diverging, we should stop the epochs from running
        if loss_counter > 0:
            print(f'Stopping the training process due to diversion with {epoch + 1} Epoch(s).')
            break

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128,256),
        nn.ReLU(),
        nn.Linear(256,64),
        nn.ReLU(),
        nn.Linear(64,5)
    )

    return model

def create_data_loaders(train_dir, eval_dir, test_dir, train_batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    # Training Image Processing (transformation, resizing, tensorization and normalization)
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(root=train_dir, transform=training_transform)
    test_data = ImageFolder(root=test_dir, transform=testing_transform)
    eval_data = ImageFolder(root=eval_dir, transform=testing_transform)

    train_loader = DataLoader(train_data, train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, test_batch_size, shuffle=True)

    return train_loader, eval_loader, test_loader

def main(args):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    print(f'Log Entry: Train batch size:{args.train_batch_size}')
    print(f'Log Entry: Test batch size:{args.test_batch_size}')
    print(f'Log Entry: Learning Rate:{args.lr}')
    print(f'Log Entry: Epochs:{args.epochs}')

    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}.')

    model=net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    train_loader, evaluation_loader, test_loader = create_data_loaders(args.train_dir, args.evaluation_dir,
                                                                       args.test_dir, args.train_batch_size,
                                                                       args.test_batch_size)

    train(model, args.epochs, train_loader, evaluation_loader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''

    test(model, test_loader, device, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = './benchmark_model'
    torch.save(model, path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        metavar='N',
        help='number of epochs for training (default:4)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        metavar='N',
        help='default learning rate for training (default:0.1)'
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16,
        metavar='N',
        help='batch size for training (default:16)'
    )

    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=8,
        metavar='N',
        help='batch size for testing (default:8)'
    )

    # Container environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--train-dir', type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--test-dir', type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument('--evaluation-dir', type=str, default=os.environ["SM_CHANNEL_EVALUATION"])

    args=parser.parse_args()
    
    main(args)
