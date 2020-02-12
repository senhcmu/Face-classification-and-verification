import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision
import time
from torchsummary import summary

from numpy.linalg import norm





class ConvBNBlock(nn.Sequential):

    def __init__(self, in_channel, out_channel, stride):
        super(ConvBNBlock, self).__init__(
            # kernel_size
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
            )


class ConvBlock(nn.Sequential):

    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
            )


class LinearInvertedResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, expand, stride):
        super(LinearInvertedResidualBlock, self).__init__()
        self.skip_connection = False

        if stride == 1 and in_channel == out_channel:
            # print("out_channel:", out_channel)
            # print("in_channel:", in_channel)
            #print(self.stride, in_channel, out_channel)
            self.skip_connection = True

        dim = round(in_channel * expand)
        self.layers = []
        if expand != 1:
            self.layers.extend([nn.Conv2d(in_channel, dim, 1, 1, 0, bias=False),
                                nn.BatchNorm2d(dim),
                                nn.ReLU6(dim)])

            self.layers.extend([nn.Conv2d(dim, dim, 3, stride, 1, groups=dim, bias=False),
                                nn.BatchNorm2d(dim),
                                nn.ReLU6(dim)])

            self.layers.extend([nn.Conv2d(dim, out_channel, 1, 1, 0, bias=False),
                                nn.BatchNorm2d(out_channel)])

        else:

            self.layers.extend([nn.Conv2d(dim, dim, 3, stride, 1, groups=dim, bias=False),
                                nn.BatchNorm2d(dim),
                                nn.ReLU6(dim)])

            self.layers.extend([nn.Conv2d(dim, out_channel, 1, 1, 0, bias=False),
                                nn.BatchNorm2d(out_channel)])



        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        # print('x shape:',x.shape)
        if self.skip_connection:
        #print('temp shape:', temp.shape)
            temp = self.layers(x)
            # print('temp size:', temp.shape)
            result = x + temp
        else:
            result = self.layers(x)
        return result





class Network(nn.Module):
    def __init__(self, num_feats, num_classes, feat_dim):
        super(Network, self).__init__()
        
        #self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        input_channel = 32
        ultimate_output_channel = 1280


        # t, c, n, s
        # self.structure = [
        #     [1, 16, 1, 1],
        #     [6, 24, 2, 2],
        #     [6, 32, 3, 2],
        #     [6, 64, 4, 2],
        #     [6, 96, 3, 1],
        #     [6, 160, 3, 2],
        #     [6, 320, 1, 1],
        # ]
        self.structure = [
            [1, 16, 1, 1],
            [6, 24, 1, 1],
            [6, 32, 2, 1],
            [6, 64, 2, 1],
            [6, 96, 1, 2],
            [6, 96, 1, 1],
            [6, 160, 1, 2],
            [6, 320, 1, 1],
        ]




        # building first layer
        # assert input_size % 32 == 0
        self.layers = []
        self.layers.append(ConvBNBlock(3, input_channel, 1))
        # building inverted residual blocks
        for t, c, n, s in self.structure:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.layers.append(LinearInvertedResidualBlock(input_channel, output_channel, t, s))
                else:
                    self.layers.append(LinearInvertedResidualBlock(input_channel, output_channel, t, 1))
                input_channel = output_channel

        self.layers.append(ConvBlock(input_channel, ultimate_output_channel))

        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(ultimate_output_channel, num_classes, bias=False)

        self.linear_closs = nn.Linear(ultimate_output_channel, feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
        # print('not yet 3')


    def forward(self, x):
        output = self.layers(x)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])

        # output = nn.Dropout(0.2, inplace=True)(output)
        
        # label_output = self.linear_label(output)
        # label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        # closs_output = self.linear_closs(output)
        # closs_output = self.relu_closs(closs_output)

        # return closs_output, label_output
        # print(output[0].tolist())
        return output.tolist()
        # return label_output


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def train(model, data_loader, valid_loader, task='Classification'):
    model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0

        train_loss = []
        accuracy = 0
        total = 0


        start_time = time.time()
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()

            feature, outputs = model(feats)
            #outputs = model(feats)


            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)
            # print("output size[2]:", outputs)
            # print(labels.long())


            #loss = criterion_label(outputs, labels.long())

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss

            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            train_loss.extend([loss.item()]*feats.size()[0])

            loss.backward()
            optimizer_label.step()

            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            
            avg_loss += loss.item()

            #if batch_num % 10 == 9:
                #print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                #avg_loss = 0.0    
                #break     


            if batch_num % 100 == 99:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/100))
                avg_loss = 0.0    
            if batch_num % 10 == 0:
                print(batch_num,'batches')
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss       

        
        PATH = "./basic_cnn06 center.pth"
            
        # torch.save(model.state_dict(), PATH)

        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_label_state_dict': optimizer_label.state_dict(),
        'optimizer_closs_state_dict': optimizer_closs.state_dict(),
        'scheduler_state_dict': scheduler,

        }, PATH )

        end_time = time.time()
         
        
        if task == 'Classification':
            print(epoch, 'Epoches')
            train_loss = np.mean(train_loss)
            train_acc = accuracy/total
         
            val_loss, val_acc = valid_classify(model, valid_loader)
            print("after test classify")
            # train_loss, train_acc = valid_classify(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
            print('Time: ',end_time - start_time, 's')

            scheduler.step(val_loss)
            # test_verify(model, valid_loader)
        # else:
        #     test_verify(model, valid_loader)
        #break


def valid_classify(model, valid_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(valid_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        #outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        # loss = criterion_label(outputs, labels.long())
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        torch.cuda.empty_cache()
        #break
        del feats
        del labels
        del loss

    model.train()
    #print(accuracy)
    return np.mean(test_loss), accuracy/total



def test_classify(model, test_loader, train_dataset):
    model.eval()
    result = []


    for batch_num, (feats, labels) in enumerate(test_loader):
        feats  = feats.to(device)
        feature, outputs = model(feats)
        #outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        result.append(int(train_dataset.classes[pred_labels]))



        
        
        # l_loss = criterion_label(outputs, labels.long())
        # c_loss = criterion_closs(feature, labels.long())
        # loss = l_loss + closs_weight * c_loss
        
        # accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        # total += len(labels)
        # test_loss.extend([loss.item()]*feats.size()[0])
        # del feats
        # del labels

    return result




def test_verify(model, test_loader):
    print('not yet 1')


    model.eval()
    result = []

    print('not yet 2')
    for batch_num, (feats, labels) in enumerate(test_loader):
        # print('not yet 3')
        feats  = feats.to(device)
        outputs = model(feats)
        # print(len(outputs))



        # outputs = model(feats)
        
        # _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        # pred_labels = pred_labels.view(-1)

        result.extend(outputs)
        del feats
        del outputs
        del labels


    test_verify_cosine_similarity(result)


# test_verification_dataset.classes[pred_labels]

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def test_verify_cosine_similarity(embedding_result):
    print(len(embedding_result))
    result = []
    contentsRead = readFile("data/test_trials_verification_student.txt")
    contentsRead = contentsRead.split('\n')

    # for i in range(len(embedding_result)-1,2):
    #     cos_sim = np.dot(embedding_result[i], embedding_result[i+1])/(norm(embedding_result[i])*norm(embedding_result[i+1]))
    #     result.append(cos_sim)

    resultOutput = ""
    resultOutput +="trial"
    resultOutput +=","
    resultOutput +="score"
    resultOutput +="\n"
    print(contentsRead[0])

    for i in range(len(contentsRead)-1):
        
        sample = contentsRead[i].split(' ')
        
        id1 = sample[0]
        id2 = sample[1]
        temp = sample[0] + ' ' + sample[1]
        resultOutput += temp
        resultOutput += ','
        pic1 = int(id1.split('.')[0])-200000
        pic2 = int(id2.split('.')[0])-200000
        # print(pic1)
        # print(pic2)

        auc = np.dot(embedding_result[pic1], embedding_result[pic2])/(norm(embedding_result[pic1])*norm(embedding_result[pic2]))

        resultOutput += str(auc)
        resultOutput += "\n"

    writeFile("label_csv.csv", resultOutput)



def readFile(path):
    with open(path, "rt") as f:
        return f.read()




# contentsRead = contentsRead.split('\n')
# result = ''
# result +="trial"
# result +=","
# result +="score"
# result +="\n"


# for sample in contentsRead:

#     sample = sample.split(' ')
#     id1 = sample[0]
#     id2 = sample[1]
#     temp = sample[0] + ' ' + sample[1]
#     result += temp
#     result += ','
#     label = sample[-1]
#     result += label 

# writeFile("true_score.csv", result)







print('will start immediately!!!')
# train_dataset = datasets.ImageFolder(root='data/train_data/medium/', 
#                                                 transform=torchvision.transforms.ToTensor())
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=150, 
#                                               shuffle=True, num_workers=8)

# dev_dataset = datasets.ImageFolder(root='data/validation_classification/medium/', 
#                                               transform=torchvision.transforms.ToTensor())
# dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=150, 
#                                             shuffle=True, num_workers=8)



# test_dataset = datasets.ImageFolder(root='data/test_classification/', 
#                                               transform=torchvision.transforms.ToTensor())

# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
#                                             shuffle=False, num_workers=8)



test_verification_dataset = datasets.ImageFolder(root='data/verification/', 
        transform=torchvision.transforms.ToTensor())

test_verification_dataloader = torch.utils.data.DataLoader(test_verification_dataset, batch_size=500,
        shuffle=False, num_workers=8)


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # print((torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)).shape)
        # print((torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()).shape)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # print(x.shape)
        # print(self.centers.t().shape)
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()


        return loss

print('after data loader!!!')





# print(train_dataset.__len__(), len(train_dataset.classes))





numEpochs = 15
num_feats = 3
# I dont know why
feat_dim = 2300
closs_weight = 0.1
# from 0.5 to 0.1
lr_cent = 0.1



learningRate = 1e-2 
weightDecay = 5e-5

#hidden_sizes = [32, 64]
num_classes = 2300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



network = Network(num_feats, num_classes, feat_dim)
network.apply(init_weights)
print('after building!')

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(num_classes, feat_dim, device)
# optimizer_label = torch.optim.Adam(network.parameters(), lr=0.01)
optimizer_label = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, 'min', patience=2)

checkpoint = torch.load('basic_cnn06 center.pth')
network.load_state_dict(checkpoint['model_state_dict'])
optimizer_label.load_state_dict(checkpoint['optimizer_label_state_dict'])

# i add the momentum
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent, momentum=0.9)
# optimizer_closs.load_state_dict(checkpoint['optimizer_closs_state_dict'])
#print(network.layers[0].weight)
#print()
# network.load_state_dict(torch.load('basic_cnn01.pt'))
#new = list(network.parameters())
# l = network.state_dict()
# print(new[0])




for state in optimizer_label.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)






#print(network.layers[0].weight)



print('enter training!')

# network.train()

network.to(device)

# summary(network, input_size=(3, 32, 32))

test_verify(network, test_verification_dataloader)
# train(network, train_dataloader, dev_dataloader)


# predictions = test_classify(network, test_dataloader, train_dataset)
# resultOutput = ""
# resultOutput +="id"
# resultOutput +=","
# resultOutput +="label"
# resultOutput +="\n"
# count = 4999
# for i in predictions:
#    count += 1
#    resultOutput += str(count)
#    resultOutput += ","
#    resultOutput += str(i)
#    resultOutput += "\n"






# writeFile("testPredictions.csv", resultOutput)


