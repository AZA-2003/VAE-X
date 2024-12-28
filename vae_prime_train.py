import torch.nn as nn
import torch as torch
import numpy as np
import time
import gc
from vae_prime import VAEX, KL_loss
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.utils
from os.path import exists
import sys

# from ignite.engine import Engine, Events
# import ignite.distributed as idist

## function to initialize model weights 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #kaiming not applicable for biases

## transformation on the image to a tensor of pixel values
input_size = 40
val_batch_size = 16
train_test_batch_size = 16

transform = transforms.Compose([ 
    transforms.Resize((input_size, input_size)),
    transforms.PILToTensor() 
]) 


## obtainind the data for trainining, validating and testing
dir = "./xray_dataset/chest_xray/"
# training
train_set = datasets.ImageFolder(f"{dir}train", transform)
train_loader = DataLoader(train_set, batch_size= train_test_batch_size, shuffle = True, drop_last=True)

# val
val_set = datasets.ImageFolder(f"{dir}val", transform)
val_loader = DataLoader(val_set, batch_size= val_batch_size, shuffle = True, drop_last=True)

# test
test_set = datasets.ImageFolder(f"{dir}test", transform)
test_loader = DataLoader(test_set, batch_size= train_test_batch_size, shuffle = True, drop_last=True)

## hyperparameters
epochs = 20
n_class = 2
norm_constant = 255
beta = 0.1
lr = 5e-4

vae_cstm = VAEX()
vae_cstm.apply(init_weights)

device = None
# check availability of gpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print ("Running on: ", device)

optimizer = torch.optim.AdamW(vae_cstm.parameters(), lr=lr)
xray_vae =  vae_cstm.to(device)

def train():
    eval_losses = []
    losses = []

    steps = epochs * len(list(enumerate(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    for epoch in range(epochs):
        ts = time.time()
        loss_tot = []
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = torch.div(inputs,255)
            # plt.imshow(torchvision.utils.make_grid(inputs,nrow=4, padding = 2).detach().cpu().numpy().transpose((1,2,0)))
            # plt.show()

            inputs = torch.reshape(inputs.to(device).type(torch.float32), \
                                    (train_test_batch_size,(input_size**2)*3))
            
            labels = labels.to(device)
            recon_x, mu, sig =  xray_vae(inputs,batch_size=train_test_batch_size)
            inputs = torch.reshape(inputs, (train_test_batch_size,3,input_size,input_size))
            
            loss = nn.functional.binary_cross_entropy(recon_x, inputs, reduction="sum")+beta*KL_loss(mu,sig)
            #print(f"train_iter {iter}, loss: {loss}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_tot.append(loss)

            if iter % 10 == 0:
                print("epoch {}, iter {}, loss: {}".format(epoch, iter, loss.item()))
                losses.append(torch.mean(torch.tensor(loss_tot)).item())
                loss_tot = []
    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        e_losses = val(epoch)
        eval_losses.append(e_losses.item())
    torch.save(xray_vae.state_dict(), f"vaex_model_w{input_size}_{beta}.pt")
    return losses, eval_losses


def val(epoch):
    xray_vae.eval()
    losses = []

    with torch.no_grad():
        for iter, (inputs,labels) in enumerate(val_loader):
            inputs = torch.reshape(inputs.to(device).type(torch.float32), \
                                    (val_batch_size,(input_size**2)*3))
            inputs = torch.div(inputs,255)
            labels = labels.to(device)
            recon_x, mu, sig =  xray_vae(inputs,batch_size=val_batch_size)

            inputs = torch.reshape(inputs, (val_batch_size,3,input_size,input_size))
            loss = nn.functional.binary_cross_entropy(recon_x, inputs,reduction="sum")+beta*KL_loss(mu,sig)
            losses.append(loss)
            #print(f"val_iter {iter}, loss: {loss.item()}")

    print(f"Val Loss at epoch: {epoch} is {torch.mean(torch.tensor(losses))}")

    xray_vae.train()
    return torch.mean(torch.tensor(losses))

def test():
    xray_vae.eval()
    random_batch = torch.randint(20, (1,5))

    with torch.no_grad():
        for iter, (inputs,labels) in enumerate(test_loader):
            inputs = torch.reshape(inputs.to(device).type(torch.float32), \
                        (train_test_batch_size,(input_size**2)*3))
            inputs = torch.div(inputs,255)
            labels = labels.to(device)
            recon_x, mu, sig =  xray_vae(inputs,batch_size=train_test_batch_size)
            inputs = torch.reshape(inputs, (train_test_batch_size,3,input_size,input_size))

            loss = nn.functional.binary_cross_entropy(recon_x, inputs,reduction="sum")+beta*KL_loss(mu,sig)

            if iter in random_batch:
                print(recon_x.shape)
                fig, axes = plt.subplots(1,2,figsize = (16,16))
                
                # plotting the originals
                axes[0].imshow(torchvision.utils.make_grid(inputs, nrow=4, padding = 2, normalize=True).cpu().numpy().transpose((1,2,0)))
                axes[0].set_title("Ground Truth")
                
                # plotting the reconstruction
                axes[1].imshow(torchvision.utils.make_grid(recon_x, nrow=4, padding = 2, normalize=True).cpu().numpy().transpose((1,2,0)))
                axes[1].set_title("Reconstructions")
            
                fig.savefig(f"imgsize_{input_size}_beta_{beta}_recon_v_groundtruth_{iter}.png")
                print("test, iter {}, loss: {}".format(iter, loss.item()))
                
    xray_vae.train()
    return 

if __name__ == "__main__":
    # show the accuracy before training
    if sys.argv[1] == "train":
        val(0)
        train_losses, eval_losses = train()
        plt.plot(np.arange(len(train_losses)),train_losses, 'r')
        plt.plot(np.arange(len(eval_losses))*int(len(train_losses)/len(eval_losses)),eval_losses, 'g')
        plt.xlabel("10 iter mark")
        plt.ylabel("avg. loss per 10 iterations")
        #plt.show()
        plt.savefig(f"imgsize_{input_size}_beta_{beta}_train_val_loss.png")

    elif sys.argv[1] == "test":
        vae_cstm.load_state_dict(torch.load(f"vaex_model_w40_0.1.pt", weights_only=True))
        xray_vae =  vae_cstm.to(device)
    
    test()
    ## add the scores of the generated images

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
