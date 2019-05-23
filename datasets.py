import torch
import scipy.io as sio
import cv2
import numpy as np
import glob
from torchvision import datasets
from torchvision import transforms
from smallnorb import *
import scipy.io as io

# currently hard code the path to bird dataset
# could be downloaded here http://www.vision.caltech.edu/visipedia/CUB-200.html
BIRD_DATASET_PATH = "./data/CUB_200_2011/images"
SVHN_DATASET_PATH = "./data/svhn/"

def Cifar10():
    batch_size=32
    use_cuda = True
    train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root='./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root='./data', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=False)

    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break

    return train_loader, test_loader, dataset_details


def AffNIST(folder = './data/training_and_validation_batches'):
    img_shape = (28,28)
    mat_contents =  sio.loadmat(folder+'/1.mat')
    nums = mat_contents['affNISTdata']

    labels = nums[0][0][5].T
    imgs = np.transpose(nums[0][0][2].reshape((40,40, 60000)), (2,0,1))
    new_imgs = []
    for index in range(len(imgs)):
        new_imgs.append(cv2.resize(imgs[index], img_shape) / imgs[index].max())
        
    imgs = np.asarray(new_imgs)

    train_images = imgs[:55000]
    train_labels = labels[:55000]

    test_images = imgs[55000:]
    test_labels = labels[55000:]

    torch_train_images = torch.from_numpy(train_images).view(-1, 1, img_shape[1],img_shape[0]).float()
    torch_train_labels = torch.from_numpy(train_labels)

    torch_test_images = torch.from_numpy(test_images).view(-1, 1, img_shape[1],img_shape[0]).float()
    torch_test_labels = torch.from_numpy(test_labels)
    print(torch_test_images.shape)
    print(torch_test_labels.shape)

    train = torch.utils.data.TensorDataset(torch_train_images, torch_train_labels.view(-1))
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=False)

    test = torch.utils.data.TensorDataset(torch_test_images, torch_test_labels.view(-1))
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
        
    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break

    return train_loader, test_loader, dataset_details


def birds():
    try:
        train_images = np.load('./data/CUB_200_2011/train_imgs.npy')
        train_labels = np.load('./data/CUB_200_2011/train_labels.npy')
        test_images = np.load('./data/CUB_200_2011/test_imgs.npy')
        test_labels = np.load('./data/CUB_200_2011/test_labels.npy')
        print("Loaded From File")
    except:
        bird_data_path = BIRD_DATASET_PATH
        img_shape = (64, 64)
        class_names = glob.glob(bird_data_path + '/*')

        def get_class(folder_name):
            class_id = int(folder_name.split('/')[-1].split('.')[0])
            return class_id

        class_names = sorted(class_names, key=get_class)

        def read_and_resize_img(img_name):
            img = cv2.imread(img_name)
            img = cv2.resize(img, img_shape) / img.max()
            return img  # a ndarray

        X = []
        y = []
        for class_name in class_names:
            class_id = get_class(class_name) - 1
            pic_names = glob.glob(class_name + '/*')
            for pic_name in pic_names:
                X.append(read_and_resize_img(pic_name))
                y.append(class_id)

        X = np.array(X)
        # make X in batch, Channel, height, width format
        X = np.transpose(X, (0, 3, 1, 2))
        assert X.shape[1] == 3
        y = np.asarray(y, dtype=np.int64)

        # random permute dataset
        order = np.arange(len(X))
        np.random.shuffle(order)

        X = X[order, :, :, :]
        y = y[order]

        train_images = X[:8000]
        train_labels = y[:8000]

        test_images = X[8000:]
        test_labels = y[8000:]

        np.save('./data/CUB_200_2011/train_imgs.npy', train_images)
        np.save('./data/CUB_200_2011/train_labels.npy', train_labels)
        np.save('./data/CUB_200_2011/test_imgs.npy', test_images)
        np.save('./data/CUB_200_2011/test_labels.npy', test_labels)

    torch_train_images = torch.from_numpy(train_images).float()
    torch_train_labels = torch.from_numpy(train_labels).long()

    torch_test_images = torch.from_numpy(test_images).float()
    torch_test_labels = torch.from_numpy(test_labels).long()
    print(torch_train_images.shape)
    print(torch_train_labels.shape)

    print(torch_test_images.shape)
    print(torch_test_labels.shape)

    train = torch.utils.data.TensorDataset(torch_train_images, torch_train_labels.view(-1))
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)

    test = torch.utils.data.TensorDataset(torch_test_images, torch_test_labels.view(-1))
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break

    return train_loader, test_loader, dataset_details


def svhn(num_train_samples=5000):
    img_shape = (32, 32)

    train_data = io.loadmat(SVHN_DATASET_PATH + "train_32x32.mat")
    test_data = io.loadmat(SVHN_DATASET_PATH + "train_32x32.mat")

    train_images = train_data["X"] / train_data["X"].max()
    train_images = np.transpose(train_images, (3, 2, 0, 1))
    train_labels = train_data["y"] - 1

    # shuffle the train set
    order = np.arange(len(train_images))
    np.random.shuffle(order)
    train_images = train_images[order, :, :, :]
    train_labels = train_labels[order]
    train_images = train_images[:num_train_samples, :, :, :]
    train_labels = train_labels[:num_train_samples]


    test_images = test_data["X"] / test_data["X"].max()
    test_images = np.transpose(test_images, (3, 2, 0, 1))
    test_labels = test_data["y"] - 1

    torch_train_images = torch.from_numpy(train_images).float()
    torch_train_labels = torch.from_numpy(train_labels).long()

    torch_test_images = torch.from_numpy(test_images).float()
    torch_test_labels = torch.from_numpy(test_labels).long()
    print(torch_test_images.shape)
    print(torch_test_labels.shape)

    train = torch.utils.data.TensorDataset(torch_train_images, torch_train_labels.view(-1))
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

    test = torch.utils.data.TensorDataset(torch_test_images, torch_test_labels.view(-1))
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break

    return train_loader, test_loader, dataset_details

def smallNORBsingle(data_root='./data/small_norb_root'):
    img_shape = (96,96)
    try:
        train_images = np.load('./data/small_norb_root/train_imgs.npy')
        train_labels = np.load('./data/small_norb_root/train_labels.npy')
        test_images = np.load('./data/small_norb_root/test_imgs.npy')
        test_labels = np.load('./data/small_norb_root/test_labels.npy')
        print("Loaded From File")
    except:
        
        dataset = SmallNORBDataset(data_root)
        train_set = dataset.data['train']
        test_set = dataset.data['test']
        
        train_images_1 = [train_set[i].image_lt for i in range(len(train_set))]
        train_images_2 = [train_set[i].image_rt for i in range(len(train_set))]
        train_images = train_images_1 + train_images_2
        train_images = np.asarray(train_images)
        
        train_labels_1 = [train_set[i].category for i in range(len(train_set))]
        train_labels_2 = [train_set[i].category for i in range(len(train_set))]
        train_labels = train_labels_1 + train_labels_2
        train_labels = np.asarray(train_labels).reshape([-1, 1])
        
        test_images_1 = [test_set[i].image_lt for i in range(len(test_set))]
        test_images_2 = [test_set[i].image_rt for i in range(len(test_set))]
        test_images = test_images_1 + test_images_2
        test_images = np.asarray(test_images)
        
        test_labels_1 = [test_set[i].category for i in range(len(test_set))]
        test_labels_2 = [test_set[i].category for i in range(len(test_set))]
        test_labels = test_labels_1 + test_labels_2
        test_labels = np.asarray(test_labels).reshape(-1, 1)
        
        np.save('./data/small_norb_root/train_imgs.npy', train_images)
        np.save('./data/small_norb_root/train_labels.npy', train_labels)
        np.save('./data/small_norb_root/test_imgs.npy', test_images)
        np.save('./data/small_norb_root/test_labels.npy', test_labels)
    
    print()
    train_labels = torch.from_numpy(train_labels).long()
    train_images = torch.from_numpy(train_images).view(-1, 1, img_shape[0],img_shape[1]).float()
    train = torch.utils.data.TensorDataset(train_images, train_labels.view(-1))
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False, num_workers=0)
        
    test_labels = torch.from_numpy(test_labels[:10000]).long()
    test_images = torch.from_numpy(test_images[:10000]).view(-1, 1, img_shape[0],img_shape[1]).float()
    test = torch.utils.data.TensorDataset(test_images, test_labels.view(-1))
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False, num_workers=0)

    print(train_images.shape)
    print(train_labels.shape)

    print(test_images.shape)
    print(test_labels.shape)

    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break
    
    return train_loader, test_loader, dataset_details

def headPose():
    def crop_face(img, x, y, width, height):
        img_height, img_width, _ = img.shape
        start_y = max(y - height, 0)
        end_y = min(y + height, img_height - 1)

        start_x = max(x - width, 0)
        end_x = min(x + width, img_width - 1)
        return img[start_y: end_y, start_x: end_x, :]

    def parse_num(num_string):
        previous_sign = 1
        parse_str = ""
        final_nums = []
        num_sign = ['-', '+']
        for c in num_string:
            if c in num_sign:
                if parse_str == "":
                    pass
                else:
                    final_nums.append(int(parse_str)*previous_sign)
                    parse_str = ""
                if c == '-':
                    previous_sign = -1
                elif c == '+':
                    previous_sign = 1
            else:
                parse_str += c
                
        return final_nums

    def create_dataset(people_data, people):
        X = []
        y = []
        for person_id in people:
            for photo_id in people_data[person_id]:
                photo_data = people_data[person_id][photo_id]   
                img = cv2.imread(photo_data['image'])
                cropped_face = crop_face(img, *photo_data['position'], *((photo_data['size']/2).astype(int)))
                cropped_face = cv2.resize(cropped_face, img_shape) / cropped_face.max()
                
                X.append(cropped_face.transpose(2, 0, 1))
                y.append(np.array([photo_data['tilt'], photo_data['pan']]))
                
                
            
        X = np.asarray(X)
        y = np.asarray(y)
        
        return X, y
    img_shape = (64, 64)

    try:
        train_images = np.load('./data/HeadPoseImageDatabase/train_imgs.npy')
        train_labels = np.load('./data/HeadPoseImageDatabase/train_labels.npy')
        test_images = np.load('./data/HeadPoseImageDatabase/test_imgs.npy')
        test_labels = np.load('./data/HeadPoseImageDatabase/test_labels.npy')
        print("Loaded From File")
    except:
        people = glob.glob('./data/HeadPoseImageDatabase/*/')
        people = [file_name.split('/')[-2] for file_name in people if 'Person' in file_name]

        file_dict = {}
        for person_id in people:
            file_dict[person_id] = {}
            picture_files = glob.glob('./data/HeadPoseImageDatabase/{}/*.jpg'.format(person_id))
            for complete_file_name in picture_files:
                new_delimeter = ';'
                file_name = complete_file_name.split('/')[-1]
                cleaned_file_name = file_name.replace('-', new_delimeter).replace('+', new_delimeter)
                base_name = cleaned_file_name.split(new_delimeter)[0]
                meta_data = file_name[len(base_name):].split('.jpg')[0]+'+'
                tilt, pan = parse_num(meta_data)
                txt_file = glob.glob('./data/HeadPoseImageDatabase/{}/{}*.txt'.format(person_id, base_name))[0]
                if base_name not in file_dict:
                    with open(txt_file, 'r') as f:
                        result = f.readlines()
                        result = [line.replace('\n', '') for line in result]
                        position = np.array((int(result[3]), int(result[4])))
                        size = np.array((int(result[5]), int(result[6])))
                        # size = np.array([64, 64])
                        file_dict[person_id][base_name] = {'image': complete_file_name, 'position':position, 'size': size,
                                                        'tilt': tilt, 'pan': pan}

        split_num = -5
        train_people = people[:split_num]
        test_people = people[split_num:]
        train_images, train_labels = create_dataset(file_dict, train_people)
        test_images, test_labels = create_dataset(file_dict, test_people)

        np.save('./data/HeadPoseImageDatabase/train_imgs.npy', train_images)
        np.save('./data/HeadPoseImageDatabase/train_labels.npy', train_labels)
        np.save('./data/HeadPoseImageDatabase/test_imgs.npy', test_images)
        np.save('./data/HeadPoseImageDatabase/test_labels.npy', test_labels)

    print(train_images.shape, train_labels.shape, img_shape)
    train_labels = torch.from_numpy(train_labels).view(-1, 2).float()
    train_images = torch.from_numpy(train_images).view(-1, 3, img_shape[0],img_shape[1]).float()
    train = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)
        
    test_labels = torch.from_numpy(test_labels).view(-1, 2).float()
    test_images = torch.from_numpy(test_images).view(-1, 3, img_shape[0],img_shape[1]).float()
    test = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    print(train_images.shape)
    print(train_labels.shape)

    print(test_images.shape)
    print(test_labels.shape)
    print(train_labels[:10])

    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break
    
    return train_loader, test_loader, dataset_details

def facialRecognition():
    def crop_face(img, x, y, width, height):
        img_height, img_width, _ = img.shape
        start_y = max(y - height, 0)
        end_y = min(y + height, img_height - 1)

        start_x = max(x - width, 0)
        end_x = min(x + width, img_width - 1)
        return img[start_y: end_y, start_x: end_x, :]

    def parse_num(num_string):
        previous_sign = 1
        parse_str = ""
        final_nums = []
        num_sign = ['-', '+']
        for c in num_string:
            if c in num_sign:
                if parse_str == "":
                    pass
                else:
                    final_nums.append(int(parse_str)*previous_sign)
                    parse_str = ""
                if c == '-':
                    previous_sign = -1
                elif c == '+':
                    previous_sign = 1
            else:
                parse_str += c
                
        return final_nums

    def create_dataset(people_data, people):
        X_train = []
        y_train = []

        X_test = []
        y_test = []
        for person_num, person_id in enumerate(people):
            for photo_id in people_data[person_id]:
                photo_data = people_data[person_id][photo_id]   
                img = cv2.imread(photo_data['image'])
                cropped_face = crop_face(img, *photo_data['position'], *((photo_data['size']/1.3).astype(int)))
                cropped_face = cv2.resize(cropped_face, img_shape) / cropped_face.max()
                if abs(photo_data['tilt']) <= 30 and abs(photo_data['pan']) <= 30:
                    X_train.append(cropped_face.transpose(2, 0, 1))
                    y_train.append(person_num)
                else:
                    X_test.append(cropped_face.transpose(2, 0, 1))
                    y_test.append(person_num)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        return X_train, y_train, X_test, y_test


    img_shape = (64, 64)

    try:
        train_images = np.load('./data/HeadPoseImageDatabase/facial_train_imgs.npy')
        train_labels = np.load('./data/HeadPoseImageDatabase/facial_train_labels.npy')
        test_images = np.load('./data/HeadPoseImageDatabase/facial_test_imgs.npy')
        test_labels = np.load('./data/HeadPoseImageDatabase/facial_test_labels.npy')
        print("Loaded From File")
    except:
        people = glob.glob('./data/HeadPoseImageDatabase/*/')
        people = [file_name.split('/')[-2] for file_name in people if 'Person' in file_name]
        people = sorted(people)

        file_dict = {}
        for person_id in people:
            file_dict[person_id] = {}
            picture_files = glob.glob('./data/HeadPoseImageDatabase/{}/*.jpg'.format(person_id))
            for complete_file_name in picture_files:
                new_delimeter = ';'
                file_name = complete_file_name.split('/')[-1]
                cleaned_file_name = file_name.replace('-', new_delimeter).replace('+', new_delimeter)
                base_name = cleaned_file_name.split(new_delimeter)[0]
                meta_data = file_name[len(base_name):].split('.jpg')[0]+'+'
                tilt, pan = parse_num(meta_data)
                txt_file = glob.glob('./data/HeadPoseImageDatabase/{}/{}*.txt'.format(person_id, base_name))[0]
                if base_name not in file_dict:
                    with open(txt_file, 'r') as f:
                        result = f.readlines()
                        result = [line.replace('\n', '') for line in result]
                        position = np.array((int(result[3]), int(result[4])))
                        size = np.array((int(result[5]), int(result[6])))
                        # size = np.array([64, 64])
                        file_dict[person_id][base_name] = {'image': complete_file_name, 'position':position, 'size': size,
                                                        'tilt': tilt, 'pan': pan}

        # images, labels = create_dataset(file_dict, people)
        # split_num = int(0.7 * len(images))
        # ind = np.arange(len(images))
        # np.random.shuffle(ind)
        # images, labels = images[ind], labels[ind]

        train_images, train_labels, test_images, test_labels= create_dataset(file_dict, people)


        np.save('./data/HeadPoseImageDatabase/facial_train_imgs.npy', train_images)
        np.save('./data/HeadPoseImageDatabase/facial_train_labels.npy', train_labels)
        np.save('./data/HeadPoseImageDatabase/facial_test_imgs.npy', test_images)
        np.save('./data/HeadPoseImageDatabase/facial_test_labels.npy', test_labels)

    print(train_images.shape, train_labels.shape, img_shape)
    train_labels = torch.from_numpy(train_labels).view(-1).long()
    train_images = torch.from_numpy(train_images).view(-1, 3, img_shape[0],img_shape[1]).float()
    train = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)
        
    test_labels = torch.from_numpy(test_labels).view(-1).long()
    test_images = torch.from_numpy(test_images).view(-1, 3, img_shape[0],img_shape[1]).float()
    test = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

    print(train_images.shape)
    print(train_labels.shape)

    print(test_images.shape)
    print(test_labels.shape)
    print(train_labels[:10])

    for img_batch, label_batch in train_loader:
        dataset_details = img_batch.shape
        break
    
    return train_loader, test_loader, dataset_details

    



    