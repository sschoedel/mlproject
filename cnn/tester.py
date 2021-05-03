from torchvision import datasets, transforms
import os
import torch
from cnn.res_cnn import ResNet
import numpy as np
from tqdm import tqdm


def test_cnn(model_path=None):

    final_model = ResNet()
    if model_path is None:
        # load the default trained model
        final_model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'cnn', 'trained_cnn'), map_location=torch.device('cpu')))
    else:
        # if model path is specified, load that model instead
        final_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    final_model.eval()
    

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
                #images, labels = data[0].to(device), data[1].to(device)
                images, labels = data
                ground_truth.extend(labels.tolist())
                outputs = final_model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted)

                progress_bar.update(1)
            
            progress_bar.close()
            print('CNN Testing Complete!')

    return ground_truth, predictions
