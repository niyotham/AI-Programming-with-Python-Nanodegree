import argparse
import json
import data_loader 
import torch
import numpy as np 
# from train import const_model
import train 
# Define command line arguments


def load_checkpoint(checkpoint, arch, hidden_units):

      # Credit to Omar Wally for availing this method to the public to find 
      # a way to convert gpu tenors to cpu 
    checkpoint_state = torch.load(checkpoint, map_location = lambda storage, loc: storage)

    class_to_idx = checkpoint_state['class_to_idx']

   
    model, optimizer, criterion = train.const_model(hidden_units, class_to_idx,arch)

    
    model.load_state_dict(checkpoint_state['state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])

    
    print(f"Loaded checkpoint => {checkpoint} with arch {checkpoint_state['arch']}, hidden units { checkpoint_state['hidden_units']} and epochs {checkpoint_state['epochs']}")
    
    return model, optimizer, criterion



def predict(image, checkpoint, topk, labels, arch, hidden_units,device, gpu=False):
    
    model, _, _ = load_checkpoint(checkpoint, arch, hidden_units)
    
    model.eval()
    
    image = data_loader.process_image(image)
    if gpu:
        input_image = torch.FloatTensor(image).cuda()
    else:
        input_image = torch.FloatTensor(image)
    model= model.to(device)
    input_image.unsqueeze_(0)
    log_ps = model(input_image)
    ps = torch.exp(log_ps)
    probs, classes = torch.topk(ps, topk)
    class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
    new_classes = []
    
    for index in classes.cpu().numpy()[0]:
        new_classes.append(class_to_idx_inv[index])
        
    return probs.cpu().detach().numpy()[0], new_classes 


def main():
    args = train.get_command_line_args()
    
    use_gpu = torch.cuda.is_available() and args.gpu
    if use_gpu:
        print("Using GPU.")
        device = torch.device('cuda')
    else:
        print("Using CPU.")
        device = torch.device('cpu')
    if(args.checkpoint,args.dir):
        probs, pred_classes = predict(args.dir, args.checkpoint,
                                     args.topk,args.labels, args.arch,
                                     args.hidden_units ,device,args.gpu)
        
    with open(args.labels, 'r') as j:
        cat_to_name = json.load(j) 

    print("---------Image Classes and their coresponding Probabilities---------")
    for i, idx in enumerate(pred_classes):
        print("Class:", cat_to_name[idx], "Probability:", probs[i])
main()