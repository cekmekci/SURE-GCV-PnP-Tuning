import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from utils import utils_mri
from utils import utils_complex_denoiser
from utils import utils_jvp

from models.network_unet import UNetRes as DRUNet


# ----------------------------------------
# Fix the random number seed
# ----------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ----------------------------------------
# Parameters of the experimental setup
# ----------------------------------------
noise_std = 0.03                     # noise level for kspace data
us_rate = 0.3                        # undersampling rate
mask_type = "RadialMask"             # type of undersampling mask

# ----------------------------------------
# Parameters of the proposed GCV-based tuning method
# ----------------------------------------
# number of iterations (K)
iter_num_max = 50                    # maximum number of iterations
iter_num_min = 1                     # minimum number of iterations
# shared parameter (lambda)
lambda_max = 1e1                     # maximum value of lambda to be tried
lambda_min = 1e-5                    # minimum value of lambda to be tried
num_samples_lambda = 5               # number of lambdas to be tried
lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num_samples_lambda).astype(np.float32)
# starting penalty parameter for the HQS algoritjm (mu^0)
mu_max = 1e1                         # maximum value of mu to be tried
mu_min = 1e-5                        # minimum value of mu to be tried
num_samples_mu = 5                   # number of mus to be tried
mus_ = np.logspace(np.log10(mu_min), np.log10(mu_max), num_samples_mu).astype(np.float32)
# parameter that scales the penalty parameters (alpha)
alpha_max = 2                        # maximum value of alpha to be tried
alpha_min = 1                        # minimum value of alpha to be tried
num_samples_alpha = 5                # number of alphas to be tried
alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_samples_alpha).astype(np.float32)

# ----------------------------------------
# Obtain the complex denoiser.
# DRUNet denoiser with pretrained weights is obtained from the following repo:
# https://github.com/cszn/DPIR
# ----------------------------------------
model_path = "./models/drunet_gray.pth"
model = DRUNet(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(torch.load(model_path))
model.eval()
model = model.to(device)
c_model = utils_complex_denoiser.ComplexDenoise(model)

# ----------------------------------------
# Prepare the MRI data
# ----------------------------------------
# obtain the mask
mask_path = './data/mri_masks/'+mask_type+'s/'+mask_type+'_'+str(us_rate)+'.npy'
mask = utils_mri.getMask(mask_path).to(device)
# obtain the MRI image (x)
im_path = './data/mri_data/1.png'
im_real = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
im = im_real + 1j * np.zeros_like(im_real)
gt = torch.from_numpy(im).to(device)
# obtain the kspace data (y)
kspace = utils_mri.A(gt, mask)
noise = noise_std * (torch.randn(*kspace.shape) + 1j * torch.randn(*kspace.shape)).to(device)
kspace = kspace + noise
# obtain the zero-filled reconstruction (A^Hy)
zf = utils_mri.Aadjoint(kspace, mask)

# ----------------------------------------
# Tune the parameters of a PnP algorithm
# ----------------------------------------
# dictionaries containing the GCV-values for different combinations of params
gcv_values = dict()
gcv_recons = dict()
# dictionaries containing the SSIM-values for different combinations of params
ssim_values = dict()
ssim_recons = dict()
zero_tensor = torch.zeros((1, 1, gt.shape[0], gt.shape[1])).float().to(device)
# random vector used to approximate the trace (b_tilde)
b = torch.randn(*kspace.shape).to(device)
# start the grid search
for mu_0 in mus_:
    for alpha in alphas:
        # --------------------------------
        # (1) specify mus
        # --------------------------------
        mus = [mu_0]
        for idxx in range(1, iter_num_max):
            mus.append(mus[idxx-1] * alpha)
        mus = torch.tensor(np.array(mus))
        for lambd in lambdas:
            # --------------------------------
            # (2) get mus and taus
            # --------------------------------
            taus = torch.sqrt(lambd / mus)
            mus, taus = mus.to(device), taus.to(device)
            # --------------------------------
            # (3) set x^0 to A^Hy
            # --------------------------------
            x = zf
            # j(x^k, y) initialized to j(x^0,y)
            j = utils_mri.Aadjoint(b, mask)
            # --------------------------------
            # (4) PnP-HQS iterations
            # --------------------------------
            for i in range(iter_num_max):
                # --------------------------------
                # (4.1) data-dependent update
                # --------------------------------
                mu = (mus[i]).float().repeat(1, 1)
                z = utils_mri.AHATIinverse(zf + mu * x, mu, mask)
                # --------------------------------
                # (4.2) denoiser step
                # --------------------------------
                with torch.no_grad():
                    x_pre = torch.unsqueeze(torch.unsqueeze(z,0),0)
                    x_pre = torch.cat((x_pre, taus[i].float().repeat(1, 1, x_pre.shape[2], x_pre.shape[3])), dim=1)
                    x = c_model(x_pre)
                # --------------------------------
                # (4.3) GCV calculation
                # --------------------------------
                # calculate the JVP
                r = utils_mri.AHATIinverse(utils_mri.Aadjoint(b, mask) + mu * torch.squeeze(j), mu, mask)
                r = torch.unsqueeze(torch.unsqueeze(r,0),0)
                r = torch.cat((r, zero_tensor), dim=1)
                j = utils_jvp.jvp_custom(c_model, x_pre, r).detach().clone()
                # calculate the residual
                Ax = utils_mri.A(x.squeeze(), mask)
                numerator = (1/(kspace.shape[0]*kspace.shape[1])) * ((((Ax - kspace).abs())**2).sum())
                # calculate the trace term
                trace_approx = torch.real((b * utils_mri.A(torch.squeeze(j), mask)).sum())
                denominator = (1 - (1/(kspace.shape[0]*kspace.shape[1])) * trace_approx)**2
                # calculate the GCV function (extention to SURE is simple)
                gcv = numerator / denominator
                # --------------------------------
                # (4.4) SSIM calculation
                # --------------------------------
                ssim_val = ssim(torch.abs(x.detach().cpu()).float().squeeze().clamp_(0, 1).numpy(), torch.abs(gt).cpu().clamp_(0, 1).numpy(), data_range=1.0)
                # --------------------------------
                # (4.5) append dictionaries
                # --------------------------------
                gcv_values[(mu_0, alpha, lambd, i+1)] = gcv.detach().cpu().numpy()
                gcv_recons[(mu_0, alpha, lambd, i+1)] = torch.abs(x.detach().cpu()).float()
                ssim_values[(mu_0, alpha, lambd, i+1)] = ssim_val
                ssim_recons[(mu_0, alpha, lambd, i+1)] = torch.abs(x.detach().cpu()).float()
                # perform the required squeezing operation
                x = torch.squeeze(x)
# find the parameters achieving minimum GCV
gcv_mu_0, gcv_alpha, gcv_lambd, gcv_num_iter = min(gcv_values, key=gcv_values.get)
# get the reconstruction
gcv_reconst = gcv_recons[(gcv_mu_0, gcv_alpha, gcv_lambd, gcv_num_iter)]

# find the parameters achieving max SSIM (oracle method)
ssim_mu_0, ssim_alpha, ssim_lambd, ssim_num_iter = max(ssim_values, key=ssim_values.get)
# get the reconstruction
ssim_reconst = ssim_recons[(ssim_mu_0, ssim_alpha, ssim_lambd, ssim_num_iter)]

# ----------------------------------------
# Print the summary and show the result
# ----------------------------------------
# print the parameters chosen by the GCV-based method and oracle method
print("GCV-based mu_0:", gcv_mu_0)
print("Oracle mu_0:", ssim_mu_0)
print("GCV-based alpha:", gcv_alpha)
print("Oracle alpha:", ssim_alpha)
print("GCV-based lambda:", gcv_lambd)
print("Oracle lambda:", ssim_lambd)
print("GCV-based number of iterations K:", gcv_num_iter)
print("Oracle number of iterations K:", ssim_num_iter)
# calculate the SSIM and print it
gcv_reconst = gcv_reconst.squeeze().clamp_(0, 1).numpy()
gt = torch.abs(gt).cpu().clamp_(0, 1).numpy()
ssim_val_gcv = ssim(gcv_reconst, gt, data_range=1.0)
print("GCV-based SSIM:", ssim_val_gcv)
ssim_reconst = ssim_reconst.squeeze().clamp_(0, 1).numpy()
ssim_val_ssim = ssim(ssim_reconst, gt, data_range=1.0)
print("Oracle SSIM:", ssim_val_ssim)
# plot the reconstructed image and the ground truth
fig, axs = plt.subplots(1, 3)
axs[0].imshow(gt)
axs[0].set_title('Ground truth')
axs[1].imshow(gcv_reconst)
axs[1].set_title('GCV-based')
axs[2].imshow(ssim_reconst)
axs[2].set_title('Oracle')
plt.show()
