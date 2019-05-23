import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

def train_net(model, trainloader, testloader, loss_function, optimizer_function, 
              num_examples = 10, num_epochs=10, display=False, validation_split=0.2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss_vals = []
    test_acc_vals = []

    validation_loss_vals = []

    train_batch_size = trainloader.batch_size
    dataset_size = len(trainloader.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=train_batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=train_batch_size,
                                                    sampler=valid_sampler)

    batch_size = testloader.batch_size
    
    total_train = min(train_batch_size * num_examples, len(train_loader.dataset))
    total_test = len(testloader.dataset)
    total_validation = split

    print("TOTAL TRAIN", total_train)
    print("TOTAL VALIDATION", total_validation)
    print("TOTAL TEST", total_test)


    for epoch in range(num_epochs):
        if display:
            print("STARTING EPOCH:", epoch)
        epoch_loss_train = 0.0
        epoch_correct_train = 0.0

        epoch_loss_validation = 0.0

        epoch_loss_test = 0.0
        epoch_correct_test = 0.0

        model.train()
        for i, batch in enumerate(train_loader):
            if i >= num_examples:
                print("STOPPING EPOCH")
                break

            input_data = Variable(batch[0].float()).to(device)
            temp_labels = Variable(batch[1]).to(device)
            # Forward + Backward + Optimize
            optimizer_function.zero_grad()
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)

            loss = loss_function(outputs, temp_labels)
            loss.backward()
            optimizer_function.step()
            
            epoch_loss_train += loss.item()
            
            if type(loss_function) == torch.nn.modules.loss.CrossEntropyLoss:
                epoch_correct_train += (predicted == temp_labels).sum().item()
            else:
                epoch_correct_train = 0

        avg_loss_train = epoch_loss_train / total_train
        avg_accuracy_train = epoch_correct_train / total_train
        

        model.eval()
        for i, batch in enumerate(validation_loader):
            input_data = Variable(batch[0].float()).to(device)
            temp_labels = Variable(batch[1]).to(device)
            # Forward + Backward + Optimize
            optimizer_function.zero_grad()
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = loss_function(outputs, temp_labels)

            epoch_loss_validation += loss.item()

        avg_loss_validation = epoch_loss_validation / total_validation
        validation_loss_vals.append(avg_loss_validation)

        for i, batch in enumerate(testloader):
            input_data = Variable(batch[0].float()).to(device)
            temp_labels = Variable(batch[1]).to(device)
            # Forward + Backward + Optimize
            optimizer_function.zero_grad()
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)
            
            loss = loss_function(outputs, temp_labels)
            epoch_loss_test += loss.item()

            if type(loss_function) == torch.nn.modules.loss.CrossEntropyLoss:
                epoch_correct_test += (predicted == temp_labels).sum().item()
            else:
                epoch_correct_test = 0

        avg_loss_test = epoch_loss_test / total_test
        avg_accuracy_test = epoch_correct_test / total_test
        
        test_loss_vals.append(avg_loss_test)
        test_acc_vals.append(avg_accuracy_test)
        print("Loss: {}, Acc: {}, Validation Loss: {}".format(avg_loss_train, 
                                                    avg_accuracy_train, avg_loss_validation))
    
    return model, test_loss_vals, test_acc_vals, validation_loss_vals

def plot_arrow_img(ax, means, orientations, img_shape, arrow_scale=2, color=(1, 0, 0), alpha=0.8):
    mean_x = means[0].cpu().data.numpy()
    mean_y = means[1].cpu().data.numpy()

    rot = orientations.cpu().data.numpy()
    arrow_start = (mean_x, img_shape[1] - mean_y)
    arrow_end = (rot[0]*arrow_scale, -1*rot[1]*arrow_scale)

    ax.arrow(arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1], 
                head_width=np.sqrt(arrow_scale/4), head_length=np.sqrt(arrow_scale/4), fc='red', ec=color, linewidth=4, alpha=alpha)


def tensor_to_numpy_img(tensor):
    tensor = tensor.cpu()
    if len(tensor.shape) == 2:
        numpy_img = tensor.data.numpy()
    if tensor.shape[1] == 3:
        numpy_img = tensor.data.squeeze(0).permute(1, 2, 0).numpy()
    elif tensor.shape[1] == 1:
        numpy_img = tensor.data.squeeze(0).squeeze(0).numpy()
    
    numpy_img += 0.01
    
    numpy_img -= numpy_img.min()
    
    numpy_img /= numpy_img.max()
    return numpy_img

def plot_img_pose(img_tensor, theta):
    print(theta.shape)
    orientation = theta[:, 0]
    position = theta[:, 2]

def plot_batch(img_batch, label_batch, pinn_model, num_examples=16):
    import matplotlib.pyplot as plt
    num_imgs, num_channels, height, width = img_batch.shape
    fig, ax = plt.subplots(2, num_examples, figsize=(200, 30))
    img_batch = img_batch.to('cpu')
    pinn_model = pinn_model.to('cpu')
    transformed_img_batch = pinn_model.correct_pose(img_batch).data
    for i in range(num_examples): 
        if num_channels == 1:
            numpy_img = img_batch[i].permute(1, 2, 0)[:, :, 0]
            numpy_transformed_img = transformed_img_batch[i].permute(1, 2, 0)[:, :, 0]
        else:
            numpy_img = img_batch[i].permute(1, 2, 0)
            numpy_transformed_img = transformed_img_batch[i].permute(1, 2, 0)
        
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[0, i].imshow(numpy_img)
        ax[1, i].imshow(numpy_transformed_img)

    plt.show()