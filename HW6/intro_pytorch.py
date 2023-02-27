import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import heapq

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    loader = None
    if training:
        #Train model
        data_set = datasets.FashionMNIST('./data',train=True,download=True,transform=custom_transform)
        loader = torch.utils.data.DataLoader(data_set, batch_size = 64)
    else:
        #Test model
        data_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(data_set, batch_size = 64, shuffle=False)
    
    return loader



def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28**2, 128), #784 Inputs (all of the images!) -> 128 Outputs
        nn.ReLU(),
        nn.Linear(128, 64), #128 Inputs -> 64 Outputs
        nn.ReLU(),
        nn.Linear(64, 10) # 64 Inputs -> 10 Outputs
    )

    return model




def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    learning_rate = 0.001
    momentum = 0.9  

    #Optimizer with Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    model.train()

    #Now we train it on each epoch with the images
    for epoch in range(T):
        epoch_steps, average_loss, correct, total = 0, 0, 0, 0

        for data, labels in train_loader:
            optimizer.zero_grad() #Zero out the current gradients

            output = model(data)
            loss = criterion(output, labels) #Counts the current accuracy

            loss.backward() #Back propogate
            optimizer.step()

            average_loss += loss.item()
            
            #Finds the number of correct predictions with our model
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            epoch_steps += 1 #This is used to calculate the number of times we calculated the loss.
       
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({(correct/total)*100:.2f}%) Loss: {average_loss/epoch_steps:.3f}")

        
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    with torch.no_grad():
        epoch_steps, average_loss, correct, total = 0, 0, 0, 0
        i = 0
        for data, labels in test_loader:
            #Same process as training, just no optimizer.
            output = model(data)
            loss = criterion(output, labels)

            average_loss += loss.item()

            #Finds the number of correct predictions with our model
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            epoch_steps += 1 #This is used to calculate the number of times we calculated the loss.

        if show_loss:
            print(f"Average loss: {(average_loss/epoch_steps):.4f}")
        print(f"Accuracy: {(correct/total)*100:.2f}%")


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    image = test_images[index]

    CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    with torch.no_grad():
        output = model(image)
        heap = list()

        prob = F.softmax(output, dim=1)
        
        for i in range(len(prob[0])):
            heapq.heappush(heap, (-prob[0][i].item(), i))
        
        #Gets the top 3
        for i in range(3):
            prediction = heapq.heappop(heap)

            print(f"{CLASS_NAMES[prediction[1]]}: {prediction[0]*100*-1:.2f}%")


if __name__ == '__main__':
    model = build_model()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader,criterion,5)
    evaluate_model(model,test_loader,criterion)
    pred_set, _ = next(iter(get_data_loader(False)))
    predict_label(model,pred_set,35)