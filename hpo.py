# Import Packages

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

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
                    loss.backwards() # Performs backward pass
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
                # Comment to train on the whole dataset:
                if samples_ran >= (0.2 * len(loader[phase].dataset)):
                    break # Stops the epoch

                # End of training and evaluation phases for an epoch

            # Convergence Check (done after each evaluation phase)
            if phase == 'evaluation':
                avg_epoch_loss = accum_loss / samples_ran
                if avg_epoch_loss < best_loss: # Loss still reducing (converging)
                    best_loss = avg_epoch_loss
                else # Loss increased (diverging)
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

    return model

def create_data_loaders(train_dir, eval_dir, test_dir, train_batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''



    args=parser.parse_args()
    
    main(args)
