#  partially borrowed from https://github.com/pratogab/batch-transforms

import numpy as np
import torch


class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.

    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pics (a list of PIL Images or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.

    """

    def __init__(self):
        self.max = 255.
        
    def __call__(self, pics):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.

        Returns:
            Tensor: Tensorized Tensor.
        """
        img = None
        if not isinstance(pics, np.ndarray):
            raise NotImplementedError
        else:
            if pics.ndim < 4:
                raise ValueError(f"batch pics should be 4 dimensional(NHWC). Got {pics.ndim} dimensions.")
            else:
                img = torch.from_numpy(pics.transpose((0, 3, 1, 2))).contiguous()

        return img.float().div_(self.max)

class PILToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    
    TODO Not haddling PIL image yet, the import thing for now is not scaled

    Convert a ``PIL Image`` to tensor.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    Args:
        pics (a list of PIL Images or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.

    """

    def __init__(self):
        pass
        
    def __call__(self, pics):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.

        Returns:
            Tensor: Tensorized Tensor.
        """
        img = None
        if not isinstance(pics, np.ndarray):
            raise NotImplementedError
        else:
            if pics.ndim < 3:
                raise ValueError(f"batch pics should be 3/4 dimensional. Got {pics.ndim} dimensions.")
            elif pics.ndim == 4:
                img = torch.from_numpy(pics.transpose((0, 3, 1, 2))).contiguous()
            else:
                pics = pics[:,None,:,:]
                img = torch.from_numpy(pics).contiguous()


        return img.float()


if __name__ == '__main__':
    pass
