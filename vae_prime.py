import torch.nn as nn
import torch as torch 

class VAEX(nn.Module):
    
    def __init__(self,input_size=40,enc_lt=120,lt_dim=20,dout=0.4):
        super().__init__()
        self.img_dim = input_size
        self.input_size = (input_size**2)*3
        # encoding
        self.dropout = nn.Dropout(dout) ## Dropout layer
        self.enc1 = nn.Linear(self.input_size,enc_lt)
        self.bn1 = nn.BatchNorm1d(enc_lt)
        self.enc2 = nn.Linear(enc_lt,enc_lt)
        self.bn2 = nn.BatchNorm1d(enc_lt)
        
        self.lt_mean = nn.Linear(enc_lt,lt_dim)
        self.lt_log_var = nn.Linear(enc_lt,lt_dim)
        
        # activations
        self.relu = nn.ReLU()
        self.soft = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        # decoding
        self.dec1 = nn.Linear(lt_dim,enc_lt)
        self.dec2 = nn.Linear(enc_lt,enc_lt)
        self.dec3 = nn.Linear(enc_lt,self.input_size)
        
    def encode(self,x):
        u1 = self.dropout(self.relu(self.bn1(self.enc1(x))))
        u2 = self.dropout(self.relu(self.bn2(self.enc2(u1))))
        
        mu = self.relu(self.lt_mean(u2))
        var = self.soft(self.lt_log_var(u2))
        
        return mu, var

    def decode(self,z,batch_size):
        y1 = self.relu(self.dec1(z))
        y2 = self.relu(self.dec2(y1))
        y3 = self.sigmoid(self.dec3(y2))
        
        y3 = torch.reshape(y3, (batch_size,3,self.img_dim,self.img_dim))
        return y3
    
    def forward(self,x,batch_size):
        mu,sig = self.encode(x)        
        epsilon = torch.randn_like(sig)
        z = mu + sig*epsilon
        x_recon = self.decode(z,batch_size)
        return x_recon, mu, sig
        
def KL_loss(mu,sigma):
    return 0.5*torch.sum(mu**2+sigma**2-(torch.log(sigma)-1))