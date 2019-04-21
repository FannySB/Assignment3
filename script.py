import numpy as np
import torch
import os
import utils
from torchvision.utils import save_image
import random
import matplotlib.pyplot as plt
from vae_svhn import VAE
# from WGAN_GP import generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = 'svhn'
input_size = 32
z_dim=100
sample_num = 100
batch_size = 64
save_dir = 'save'
samples_dir = 'samples'
log_dir = 'log'
gpu_mode = True
lambda_ = 10
n_critic = 5 
sample_z = torch.randn((batch_size, z_dim)).to(device)


def visualize_sample(z_latent, model_name, epsilon = 0, dim_eps = 5):
    dir_recon = 'latent_z'
    if not os.path.exists(samples_dir + '/' + model_name):
        os.makedirs(samples_dir + '/' + model_name)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    z_samples = z_latent.view(-1, z_dim)

    # pdb.set_trace()
    if epsilon != 0:
        z_trans = torch.transpose(z_latent, 0, 1)
        if dim_eps != 0:
            z_before = z_trans[:][:dim_eps]
        z_eps = z_trans[:][dim_eps] + epsilon
        if dim_eps != z_dim - 1:
            z_after = z_trans[:][dim_eps+1:]
        
        if dim_eps == 0:
            z_final = torch.cat((z_eps.view(1, -1), z_after), 0)
        elif dim_eps == z_dim - 1:
            z_final = torch.cat((z_before, z_eps.view(1, -1)), 0)
        else:
            z_final = torch.cat((z_before, z_eps.view(1, -1), z_after), 0)
        z_samples = torch.transpose(z_final, 0, 1)

    samples = model.decode(z_samples)
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2

    if epsilon != 0:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(epsilon) + '_dim' + str(dim_eps) + '.png')
    else:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '.png')

    return z_samples

def load(model_name):
    save_path = os.path.join(save_dir, dataset, model_name)
    model.load_state_dict(torch.load(os.path.join(save_path, model_name + '.pkl')))


if __name__ == "__main__":

    model_name = 'vae'

    if model_name == 'vae':
        model = VAE().to(device)
    elif model_name == 'WGAN_GP_G':
        model = generator()
    
    load(model_name)
    z_latent = torch.randn((batch_size, z_dim)).to(device)
    # z_latent = torch.randn((2, 10)).to(device)

    # visualize_sample(z_latent, model_name)
    # visualize_sample(z_latent, model_name, 10000, 0)

    # z_latent = torch.randn((batch_size, z_dim)).to(device)
    # z_latent = torch.randn((2, 10)).to(device)

    # visualize_sample(z_latent, model_name, 2)
    # n1 = visualize_sample(z_latent, model_name, 1, 1)
    # n2 = visualize_sample(n1, model_name, 1, 2)
    # n3 = visualize_sample(n2, model_name, 1, 3)
    # n4 = visualize_sample(n3, model_name, 1, 4)
    # n5 = visualize_sample(n4, model_name, 1, 5)
    # n6 = visualize_sample(n5, model_name, 1, 6)
    #
    # for dim_eps in range(z_dim):
    #     tmp = visualize_sample(z_latent, model_name, epsilon, dim_eps)

    def testrec(input, dim_eps):

        if dim_eps < 99:
            output = visualize_sample(input, model_name, 10, dim_eps)
            return (testrec(output, dim_eps+1))
        else:
            return visualize_sample(input, model_name, 10, dim_eps)

    final = testrec(z_latent, 0)
    print(final)


    # epsilon = 10000
    # for dim_eps in range(z_dim):
    #     visualize_sample(z_latent, model_name, epsilon, dim_eps)



