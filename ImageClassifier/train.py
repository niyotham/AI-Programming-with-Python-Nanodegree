import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import argparse

import torchvision.models as models
from PIL import Image
import sys
import time
import copy
from collections import OrderedDict

import data_loader
# from data_loader import load_data
# import predict

def get_command_line_args():
    
    parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the           probability of that name.')
    parser.add_argument('--dir', type=str, default='flowers',help='Images Folder')
    parser.add_argument('--topk', type=int, help='Top classes to return', default=5)
    #parser.add_argument('--checkpoint', type=str, default='.',help='Save trained model to file')
    parser.add_argument('--checkpoint', type=str, help='Saved Checkpoint') 
    parser.add_argument('--gpu', default='False',action='store_true', help='Where to use gpu or cpu')
    parser.add_argument('--epoch', type=int, default = '10' , help='amount of times the model will train')
    parser.add_argument('--labels', type=str, help='file for label names',default='paind-project/cat_to_name.json')
    parser.add_argument('--learning_rate', type=float, default='0.001',help='Learning rate')
    parser.add_argument('--arch', type=str, default='vgg16', help='chosen model')
    parser.add_argument('--hidden_units', type=int, default='512', help='hidden units for the model')
   
 
    args = parser.parse_args()
    
    return args

def const_model(hidden_units,class_to_idx,architecture='vgg16',learning_rate = 0.001):
    """function to provide different model architecture """

    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_features    
    else:
        raise  TypeError("The architecture specified is not supported")
        
    # Freezing parameters so we don't backpropagate through them. 
    for param in model.parameters():
        param.require_grad = False
        
    
    output_size =102
    
    # define the classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size,hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(hidden_units,output_size)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    
    model.classifier = classifier
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    model.class_to_idx =  class_to_idx 
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion 
                                            
                                            
                                            
# define a function to train the model                                          
def train_model(image_datasets, architecture, 
                hidden_units, epochs,                                                                                       learning_rate, 
                class_to_idx,gpu=True,
                checkpoint=''):
    
    model, optimizer, criterion = const_model(hidden_units,class_to_idx,
                                                  architecture, learning_rate)                                  
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()                                   
    else:
       device = torch.device("cpu") 
                                            
    print('Network architecture:', architecture)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)
                                            
    # Use gpu if selected and available thanks to omar for availing this method on.
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")                                
    sys.stdout.write('Training')    
    
    steps = 0
    acc = 0
    epochs = 10
    print_every = 40
    time_start =time.time()
    
    # load the dataloaders
    _, trainloader, validloader, _,_ =data_loader.load_data(image_datasets)                                           
    for e in range(epochs):
        training_loss = 0
        valid_loss = 0                                     
        for i, (image, label) in enumerate(trainloader):
            model.train()
            steps += 1 
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            log_ps = model.forward(image)
            loss = criterion(log_ps,label)
            loss.backward()
            optimizer.step()                                
            if i % 5 == 0:
               sys.stdout.write('.')
               sys.stdout.flush()
               running_loss +=  loss.item() 
                                            
            if steps % print_every == 0:
                for i, (image, label) in enumerate(validloader):
                    model.eval()                                
                    image, label = image.to(device), label.to(device)

                    log_ps = model.forward(image)

                    valid_loss += criterion(log_ps, label).item()

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                            "Loss: {:.3f}".format(training_loss/print_every), 
                            "Validation Loss: {:.3f}".format(valid_loss/len(validloader)),
                            "Valid accuracy: {:.3f}".format(valid_loss/len(validloader)))

                    print()
                    sys.stdout.write("Training")
                    running_loss = 0 
                    running_loss1 = 0
    time_end = time.time()- time_start
    
    print("Training completed!")
    
    print("\n** Total Elapsed Runtime:",
          str(int((time_end/3600)))+":"+str(int((time_end%3600)/60))+":"
          +str(int((time_end%3600)%60)) )
                                            
    #x = test_model(path,data_load,model)
                                            
    return model, optimizer, criterion

def test_model(image_datasets, data_load,model):
    _, _, _,testloader,_ = data_loader.load_data(image_datasets)                                       
    print('Testing start')
                                            
    model.to('cuda')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (100 * correct / total))
                                            
                                            
def save_checkpoint(arch, learning_rate, hidden_units,
                    epochs, save_path, optimizer,criterion,class_to_idx):
    
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': arch.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'class_to_idx': class_to_idx
    }

    torch.save(state, save_path)
    
    print("Checkpoint saved to {}".format(save_path)) 

def main():     
    args = get_command_line_args()
   
    _, _, _, _,class_to_idx = data_loader.load_data(args.dir)
#     print(trainloader)
    model,optimizer,criterion =  train_model(args.dir,args.arch,  args.hidden_units, 
                   args.epoch, args.learning_rate,class_to_idx,args.gpu,args.checkpoint )
                                            
# check  if a user did input checkpoint destination or no as Omar advised   
    if len(args.checkpoint) == 0:
        args.checkpoint = args.checkpoint + 'checkpoint.pth'
    else:
        args.checkpoint = args.checkpoint + '/checkpoint.pth'
                                            
    save_checkpoint(model,args.learning_rate,args.hidden_units,
                   args.epoch,args.checkpoint,optimizer,criterion,class_to_idx)
                                            
#advise to use this method to avoid training when calling const_model function
#  from predict.py
if __name__ == "__main__":
    main()                                            