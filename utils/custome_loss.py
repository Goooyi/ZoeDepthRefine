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

class MaskedMAELoss(nn.Module):
    def __init__(self, lambd, name=None):
        super(MaskedMAELoss, self).__init__()
        self.__name__ = "maskmae_loss"
        self.lambd = lambd
        self.name = name
    
    def forward(self, input, target):
        # Calculate the masked MAE loss
        mae_loss = torch.abs(input[:,0,:,:] - target[:,0,:,:])
        mask = torch.where(target[:,1,:,:] != 0, self.lambd, 1.0-self.lambd)
        mae_loss *= mask
        if abs(self.lambd - 1) < 1E-5:
            mae_loss = torch.sum(mae_loss) / torch.sum(mask)
        else:
            mae_loss = torch.mean(mae_loss)
    
        return mae_loss

class MaskedRMSELoss(nn.Module):
    def __init__(self, lambd, name=None):
        super(MaskedRMSELoss, self).__init__()
        self.__name__ = "maskrmse_loss"
        self.lambd = lambd
        self.name = name
    
    def forward(self, input, target):
        # Calculate the masked RMSE loss
        rmse_loss = torch.sqrt(torch.pow(input[:,0,:,:] - target[:,0,:,:], 2))
        mask = torch.where(target[:,1,:,:] != 0, self.lambd, 1.0-self.lambd)
        rmse_loss *= mask
        if abs(self.lambd - 1) < 1E-5:
            rmse_loss = torch.sum(rmse_loss) / torch.sum(mask)
        else:
            rmse_loss = torch.mean(rmse_loss)
    
        return rmse_loss

class MaskedMSRLoss(nn.Module):
    # Calculates the mean square relative error between input and target
    def __init__(self, lambd, name=None, epsilon=1e-9):
        super(MaskedMSRLoss, self).__init__()
        self.__name__ = "maskmsr_loss"
        self.lambd = lambd
        self.name = name
        self.epsilon = epsilon

    def forward(self, input, target):
        # Calculate the masked MSR loss
        msr_loss = (input[:,0,:,:] - target[:,0,:,:])**2 / (target[:,0,:,:]**2 + self.epsilon)
        mask = torch.where(target[:,1,:,:] != 0, self.lambd, 1.0-self.lambd)
        msr_loss *= mask
        # filter out the gt of value 0
        msr_loss = torch.where(target[:,0,:,:] <= 0.2, 0, msr_loss)
        if abs(self.lambd - 1) < 1E-5:
            msr_loss = torch.sum(msr_loss) / torch.sum(mask)
        else:
            msr_loss = torch.mean(msr_loss)

        return msr_loss

class MaskedRMSLELoss(nn.Module):
    def __init__(self, lambd, name=None):
        super(MaskedRMSLELoss, self).__init__()
        self.__name__ = "maskrmse_loss"
        self.lambd = lambd
        self.name = name
    
    def forward(self, input, target):
        # Calculate the masked Root Mean Squared Logarithmic Error
        mask = torch.where(target[:,1,:,:] != 0, self.lambd, 1.0-self.lambd)
        input = torch.log(input[:,0,:,:] + 1)
        target = torch.log(target[:,0,:,:] + 1)
        rmse_loss = torch.pow(input - target, 2)
        rmse_loss *= mask
        if abs(self.lambd - 1) < 1E-5:
            rmse_loss = torch.sqrt(torch.sum(rmse_loss) / torch.sum(mask))
        else:
            rmse_loss = torch.sqrt(torch.mean(rmse_loss))
    
        return rmse_loss

if __name__ == '__main__':
	pass