from torchvision import datasets, transforms
import os
import torch
from cnn.res_cnn import ResNet
import numpy as np
from tqdm import tqdm
import time

def test_cnn(model_path=None, gpu=False):

    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else: 
            print('CUDA GPU not found! Moving to CPU...')
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")

    print(f'running on {device}')
    
    start_time = time.time()
    print('Loading pretrained cnn model')
    final_model = ResNet()
    final_model.to(device)
    if model_path is None:
        # load the default trained model
        final_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'cnn', 'trained_cnn.pth'), map_location=torch.device('cpu')))
    else:
        # if model path is specified, load that model instead
        final_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    final_model.eval()
    print(f'Loading model took {time.time() - start_time} seconds.')

    # specify hyperparams
    BATCH_SIZE = 4
    WORKERS = 0

    # specify what transformations to apply on each image
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor(),
        transforms.CenterCrop(100)
    ])

    # test the final model
    testset = datasets.ImageFolder(os.path.join(os.getcwd(), 'Test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS)
    
    ground_truth = []
    predictions = []
    with torch.no_grad():
        with tqdm(total=len(testloader), desc="Testing...") as progress_bar:

            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                ground_truth.extend(labels.tolist())
                outputs = final_model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted)

                progress_bar.update(1)
            
            progress_bar.close()
            print('CNN Testing Complete!')

    return ground_truth, predictions
