import torch

class ComplexDenoise(torch.nn.Module):
    """Complex denoiser class."""

    def __init__(self, denoiser):
        super(ComplexDenoise, self).__init__()
        self.denoiser = denoiser

    def forward(self, z):
        # separate the real and imaginary parts of the input
        z_real = torch.unsqueeze(torch.real(z[:,0,:,:]), 0).float()
        z_imag = torch.unsqueeze(torch.imag(z[:,0,:,:]), 0).float()
        # get the noise level
        noise_level = torch.unsqueeze(torch.real(z[:,1,:,:]), 0).float()
        # denoise real and imaginary parts separately
        x_real_pre = torch.cat((z_real, noise_level), dim=1)
        x_imag_pre = torch.cat((z_imag, noise_level), dim=1)
        x_real_temp = self.denoiser(x_real_pre)
        x_imag_temp = self.denoiser(x_imag_pre)
        x_temp = x_real_temp + 1j * x_imag_temp
        return x_temp
