import torch
import torch.nn.functional as F
import torch.nn as nn

class MaskedMSELoss_resized(nn.Module):
    def __init__(self):
        super(MaskedMSELoss_resized, self).__init__()
        self.__name__ = "maskmse_loss_resized"
    
    def forward(self, input, target):
        # Calculate the masked MSE loss, resize to original first
        input_intp = F.interpolate(input, size=(480, 640), mode="bilinear")
        mse_loss = (input_intp[:,0,:,:] - target[:,0,:,:]) ** 2
        mse_loss *= target[:,1,:,:]
        mse_loss = torch.mean(mse_loss)
    
        return mse_loss


class MaskedMSELoss(nn.Module):
    def __init__(self, lambd, name=None):
        super(MaskedMSELoss, self).__init__()
        self.__name__ = "maskmse_loss"
        self.lambd = lambd
        self.name = name
    
    def forward(self, input, target):
        # Calculate the masked MSE loss
        mse_loss = (input[:,0,:,:] - target[:,0,:,:]) ** 2
        mask = torch.where(target[:,1,:,:] != 0, self.lambd, 1.0-self.lambd)
        mse_loss *= mask
        if abs(self.lambd - 1) < 1E-5:
            mse_loss = torch.sum(mse_loss) / torch.sum(mask)
        else:
            mse_loss = torch.mean(mse_loss)
    
        return mse_loss

if __name__ == '__main__':
	pass