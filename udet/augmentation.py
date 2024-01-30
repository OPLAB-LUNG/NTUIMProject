import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import torchvision.transforms as transforms

class DADataset(Dataset):
    def __init__(self, ct_scan, files=None):
        super(DADataset, self).__init__()
        ct_scan[ct_scan < -1000] = -1000
        ct_scan[ct_scan > 1000] = 1000
        
        self.files = ct_scan

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        slice_num = idx
        if idx > (len(self.files) - 4):
            im = torch.from_numpy(np.stack((self.files[idx], self.files[idx], self.files[idx], self.files[idx]), axis=0)).float()
        else: 
            im = torch.from_numpy(np.stack((self.files[idx], self.files[idx + 1], self.files[idx + 2], self.files[idx + 3]), axis=0)).float()
        return im, slice_num
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_conv1 = self.double_conv(4, 64)
        self.down_conv2 = self.double_conv(64, 128)
        self.down_conv3 = self.double_conv(128, 256)
        self.down_conv4 = self.double_conv(256, 512)
        self.up_conv1 = self.double_conv(512 + 256, 256)
        self.up_conv2 = self.double_conv(256 + 128, 128)
        self.up_conv3 = self.double_conv(128 + 64, 64)
        self.up_conv4 = nn.Conv2d(64, 1, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downward path
        x1 = self.down_conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down_conv4(x6)

        # Upward path
        x = self.upsample(x7)
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv1(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv3(x)
        x = self.up_conv4(x)
        
        return x.squeeze(0)
    
def init_augment_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "cuda" only when GPUs are available.
    model_best = UNet().to(device)
    _exp_name = "all_without_4slice"
    # model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    return model_best, device


def augment(ct_scan, model_best, device):

    # The number of batch size.
    batch_size = 1
    test_set =  DADataset(ct_scan)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    slicesize = len(ct_scan)
    
    with torch.no_grad():
        pred_mat = []
        orig_mat = []
        # for im, slice_num in tqdm(test_loader, desc="Prediction"):
        for im, slice_num in test_loader:
            orig_mat.append(im[:, 0, :, :])
            if slice_num[0].item() == 0:
                pred_mat.append(im[:, 0, :, :])
            if slice_num[0].item() <= float(slicesize - 4):
                test_pred = model_best(im.to(device))
                pred_mat.append(test_pred)
            elif slice_num[0].item() == float(slicesize) - 3:
                pass
            elif slice_num[0].item() == float(slicesize - 1):
                pass
            else:
                pred_mat.append(im[:, 0, :, :])
        aug_len = 2 * slicesize - 1
        aug_data = torch.empty((aug_len, 512 * 512), dtype=torch.int64)
        
        orig_np = np.empty((slicesize, 512, 512))
        for index, tensor in enumerate(orig_mat):
            orig_np[index] = tensor.cpu().numpy().reshape(512, 512)
            
        pred_np = np.empty((slicesize - 1, 512, 512))
        for index, tensor in enumerate(pred_mat):
            pred_np[index] = tensor.cpu().numpy().reshape(512, 512)
            
        final = np.empty((orig_np.shape[0] + pred_np.shape[0], 512, 512))
        final[::2] = orig_np
        final[1::2] = pred_np  
            
    return final