import numpy as np
import torch
import os
import utils
from torchvision.utils import save_image
import random
import imageio
import matplotlib.pyplot as plt
from vae_svhn import VAE
import pdb
from WGAN_GP import generator

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



def visualize_all_sample(z_latent, model_name, count):
    dir_recon = 'latent_z'
    path = 'fid_samples' + '/' + model_name
    if not os.path.exists(path):
        os.makedirs(path)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    z_samples = z_latent.view(-1, z_dim)
    samples = model.decode(z_samples)

    for sample in samples:
        count += 1
        save_image(sample, path + '/' + model_name + '_' + str(count) + '.png')
    return samples

    
def visualize_gan_all_sample(z_latent, model_name, count):
    dir_recon = 'latent_z'
    path = 'fid_samples' + '/' + model_name + '/samples' 
    if not os.path.exists(path):
        os.makedirs(path)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    z_samples = z_latent.view(-1, z_dim)
    samples = model(z_samples.cpu())

    for sample in samples:
        count += 1
        save_image(sample, path + '/' + model_name + '_' + str(count) + '.png')
    return samples


def visualize_sample(z_latent, model_name, epsilon = 0, dim_eps = 5):
    dir_recon = 'latent_z'
    path = samples_dir + '/' + model_name
    if not os.path.exists(path):
        os.makedirs(path)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    z_samples = z_latent.view(-1, z_dim)

    z_samples = torch.transpose(z_samples, 0, 1)
    z_samples[dim_eps] = z_samples[dim_eps] + epsilon
    z_samples = torch.transpose(z_samples, 0, 1)

    samples = model.decode(z_samples)
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2

    if epsilon != 0:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(epsilon) + '_dim' + str(dim_eps) + '.png')
        # save_image(samples_[0], samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(epsilon) + '_dim' + str(dim_eps) + '.png')
    else:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        path + '/' + model_name + '.png')

    return samples

def visualize_gan_sample(z_latent, model_name, epsilon = 0, dim_eps = 5):
    dir_recon = 'latent_z'
    path = samples_dir + '/' + model_name
    if not os.path.exists(path):
        os.makedirs(path)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    z_samples = z_latent.view(-1, z_dim)

    z_samples = torch.transpose(z_samples, 0, 1)
    z_samples[dim_eps] = z_samples[dim_eps] + epsilon
    z_samples = torch.transpose(z_samples, 0, 1)

    samples = model(z_samples.cpu())
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2

    if epsilon != 0:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(epsilon) + '_dim' + str(dim_eps) + '.png')
        # save_image(samples_[0], samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(epsilon) + '_dim' + str(dim_eps) + '.png')
    else:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        path + '/' + model_name + '.png')

    return samples


def generate_sample(z_latent, model_name, epsilon = 0, dim_eps = 5):
    dir_recon = 'latent_z'
    if not os.path.exists(samples_dir + '/' + model_name):
        os.makedirs(samples_dir + '/' + model_name)

    z_samples = z_latent.view(-1, z_dim)

    z_samples = torch.transpose(z_samples, 0, 1)
    z_samples[dim_eps] = z_samples[dim_eps] + epsilon
    z_samples = torch.transpose(z_samples, 0, 1)

    samples = model.decode(z_samples)
    # normalize = transform.normalize(mean(-1, -1, -1), std(2, 2, 2))
    # sam = normalize(samples)
    samples = samples[7].view(1, 3, input_size, input_size)
    samples = samples.cpu().data.numpy() #.transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2

    return samples

def generate_gan_sample(z_latent, model_name, epsilon = 0, dim_eps = 5):
    dir_recon = 'latent_z'
    if not os.path.exists(samples_dir + '/' + model_name):
        os.makedirs(samples_dir + '/' + model_name)

    z_samples = z_latent.view(-1, z_dim)

    z_samples = torch.transpose(z_samples, 0, 1)
    z_samples[dim_eps] = z_samples[dim_eps] + epsilon
    z_samples = torch.transpose(z_samples, 0, 1)

    samples = model(z_samples.cpu())
    samples = samples[7].view(1, 3, input_size, input_size)
    samples = samples.cpu().data.numpy() #.transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2

    return samples

def img_interpolate(img1, img2, alpha):
    img1 = img1.flatten()
    img2 = img2.flatten()

    out = alpha * (img1) + (1 - alpha) * (img2)

    out = out.reshape(64, input_size, input_size, 3)

    path = 'results/svhn/vae/latent/interpol.png'

    utils.save_images(out[:8 * 8, :, :, :], [8, 8], path)

    return out

def z_interpolate(z_1, z_2, alpha):
    z_1 = z_1.flatten()
    z_2 = z_2.flatten()

    out = alpha * (z_1) + (1 - alpha) * (z_2)

    out = out.reshape(64, input_size, input_size, 3)

    path = 'results/svhn/vae/latent/interpol_z.png'

    utils.save_images(out[:8 * 8, :, :, :], [8, 8], path)

    return out


def load(model_name):
    save_path = os.path.join(save_dir, dataset, model_name)
    model.load_state_dict(torch.load(os.path.join(save_path, model_name + '.pkl')))

if __name__ == "__main__":

    print('VAE')
    model_name = 'vae'
    model = VAE().to(device)
    
    load(model_name)
    z_latent = torch.randn((batch_size, z_dim)).to(device)
    visualize_sample(z_latent, model_name)

    print('print samples for #3 quantitative')
    for i in range(15):
        z_latent = torch.randn((batch_size, z_dim)).to(device)
        visualize_all_sample(z_latent, model_name, i*64)

    print('print samples with epsilon')
    image_frame_dim = int(np.floor(np.sqrt(z_dim)))
    epsilon = [0.9]
    for eps in epsilon:
        samples = np.ndarray((z_dim, 3, input_size, input_size))
        for dim_eps in range(z_dim):
            new_sample = generate_sample(z_latent, model_name, eps, dim_eps)
            samples[dim_eps] = new_sample
    
        samples = samples[:image_frame_dim * image_frame_dim]
        samples = samples.transpose(0, 2, 3, 1)

        utils.save_images(samples, [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(eps) + '.png')


    
    print('WGAN_GP_G')
    model_name = 'WGAN_GP_G'
    input_size = 28
    model = generator(input_dim=z_dim, output_dim=3, input_size=28)

    load(model_name)
    model.eval()
    
    z_latent = torch.randn((batch_size, z_dim)).to(device)
    visualize_gan_sample(z_latent, model_name)
    print('print samples for #3 quantitative')
    for i in range(15):
        z_latent = torch.randn((batch_size, z_dim)).to(device)
        visualize_gan_all_sample(z_latent, model_name, i*64)

    print('print samples with epsilon')
    for eps in epsilon:
        samples = np.ndarray((z_dim, 3, input_size, input_size))
        for dim_eps in range(z_dim):
            # pdb.set_trace()
            new_sample = generate_gan_sample(z_latent, model_name, eps, dim_eps)
            samples[dim_eps] = new_sample
    
        # pdb.set_trace()
        samples = samples[:image_frame_dim * image_frame_dim]
        samples = samples.transpose(0, 2, 3, 1)

        utils.save_images(samples, [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(eps) + '.png')




    ### Sanity check
    # def testrec(input, dim_eps):
    #
    #     if dim_eps < 99:
    #         output = visualize_sample(input, model_name, 10, dim_eps)
    #         return (testrec(output, dim_eps+1))
    #     else:
    #         return visualize_sample(input, model_name, 10, dim_eps)
    #
    # final = testrec(z_latent, 0)
    # print(final)


    # epsilon = 10000
    # for dim_eps in range(z_dim):
    #     visualize_sample(z_latent, model_name, epsilon, dim_eps)



