from model import DiffusionModel
from unet import Unet

def main():
    sampling_timesteps = None

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()
    diffusion = DiffusionModel(
        model,
        timesteps=1000,   # number of timesteps
        sampling_timesteps=sampling_timesteps,
    ).cuda()

if __name__ == "__main__":
    main()