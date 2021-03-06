import numpy as np
import torch
import os
import utils
from torchvision.utils import save_image, make_grid
import random
import imageio
import matplotlib.pyplot as plt
from vae_svhn import VAE
import pdb
from WGAN_GP import generator
from torchvision import transforms
from torch.nn import functional as F
from torch.autograd import Variable

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
mu = (-1, -1, -1)
var = (2, 2, 2)


def visualize_all_sample(z_latent, model_name, count):
    dir_recon = 'latent_z'
    path = 'fid_samples' + '/' + model_name + '/samples' 
    if not os.path.exists(path):
        os.makedirs(path)

    tot_num_samples = min(sample_num, batch_size)
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    z_samples = z_latent.view(-1, z_dim)
    samples = model.decode(z_samples)
    
    for sample in samples:
        count += 1
        if count>1000: break
        # pdb.set_trace()
        sample = sample.view(1, 3, input_size, input_size).cpu().data.numpy().transpose(0, 2, 3, 1)
        utils.save_images(sample, [1, 1], path + '/' + model_name + '_' + str(count) + '.png')
        # save_image(sample, path + '/' + model_name + '_' + str(count) + '.png')

    
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
        if count>1000: break
        sample = sample.view(1, 3, input_size, input_size).cpu().data.numpy().transpose(0, 2, 3, 1)
        utils.save_images(sample, [1, 1], path + '/' + model_name + '_' + str(count) + '.png')


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

    # samples = (samples + 1) / 2

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


    if epsilon != 0:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        samples_dir + '/' + model_name + '/' + model_name + '_eps' + str(epsilon) + '_dim' + str(dim_eps) + '.png')
    else:
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        path + '/' + model_name + '.png')

    return samples



def z_gan_interpolate(z_1, z_2, alpha):
    if not os.path.exists('samples/WGAN_GP_G/latent/'):
        os.makedirs('samples/WGAN_GP_G/latent/')
    z_1 = z_1.flatten()
    z_2 = z_2.flatten()
    samples1 = []
    samples2 = []
    samples3 = []
    samples4 = []
    samples5 = []
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for alpha in alphas:
        out = alpha * (z_1) + (1 - alpha) * (z_2)
        out = out.reshape(64, z_dim)
        sample = model(out.cpu())
        samples1.append(sample[0].cpu().data.numpy())
        samples2.append(sample[1].cpu().data.numpy())
        samples3.append(sample[2].cpu().data.numpy())
        samples4.append(sample[3].cpu().data.numpy())
        samples5.append(sample[4].cpu().data.numpy())

    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    samples3 = np.array(samples3)
    samples4 = np.array(samples4)
    samples5 = np.array(samples5)

    image_frame_dim = len(alphas)
    path = 'samples/WGAN_GP_G/latent/interpol_z_1.png'
    print_samples = samples1[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_z_2.png'
    print_samples = samples2[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_z_3.png'
    print_samples = samples3[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_z_4.png'
    print_samples = samples4[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_z_5.png'
    print_samples = samples5[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)

    return out



def img_gan_interpolate(z_1, z_2, alpha):
    if not os.path.exists('samples/WGAN_GP_G/latent/'):
        os.makedirs('samples/WGAN_GP_G/latent/')
    samples1 = []
    samples2 = []
    samples3 = []
    samples4 = []
    samples5 = []
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    sample1 = model(z_1.cpu())
    sample2 = model(z_2.cpu())
    for alpha in alphas:
        out = alpha * (sample1) + (1 - alpha) * (sample2)
        samples1.append(out[0].cpu().data.numpy())
        samples2.append(out[1].cpu().data.numpy())
        samples3.append(out[2].cpu().data.numpy())
        samples4.append(out[3].cpu().data.numpy())
        samples5.append(out[4].cpu().data.numpy())

    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    samples3 = np.array(samples3)
    samples4 = np.array(samples4)
    samples5 = np.array(samples5)

    image_frame_dim = len(alphas)
    # pdb.set_trace()
    path = 'samples/WGAN_GP_G/latent/interpol_1.png'
    print_samples = samples1[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_2.png'
    print_samples = samples2[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_3.png'
    print_samples = samples3[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_4.png'
    print_samples = samples4[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/WGAN_GP_G/latent/interpol_5.png'
    print_samples = samples5[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)

    return out


def z_interpolate(z_1, z_2, alpha):
    if not os.path.exists('samples/vae/latent/'):
        os.makedirs('samples/vae/latent/')
    z_1 = z_1.flatten()
    z_2 = z_2.flatten()
    samples1 = []
    samples2 = []
    samples3 = []
    samples4 = []
    samples5 = []
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for alpha in alphas:
        out = alpha * (z_1) + (1 - alpha) * (z_2)
        out = out.reshape(64, z_dim)
        sample = model.decode(out)
        samples1.append(sample[0].cpu().data.numpy())
        samples2.append(sample[1].cpu().data.numpy())
        samples3.append(sample[2].cpu().data.numpy())
        samples4.append(sample[3].cpu().data.numpy())
        samples5.append(sample[4].cpu().data.numpy())

    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    samples3 = np.array(samples3)
    samples4 = np.array(samples4)
    samples5 = np.array(samples5)

    image_frame_dim = len(alphas)
    # pdb.set_trace()
    path = 'samples/vae/latent/interpol_z_1.png'
    print_samples = samples1[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_z_2.png'
    print_samples = samples2[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_z_3.png'
    print_samples = samples3[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_z_4.png'
    print_samples = samples4[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_z_5.png'
    print_samples = samples5[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)

    return out


def img_interpolate(z_1, z_2, alpha):
    if not os.path.exists('samples/vae/latent/'):
        os.makedirs('samples/vae/latent/')
    # z_1 = z_1.flatten()
    # z_2 = z_2.flatten()
    samples1 = []
    samples2 = []
    samples3 = []
    samples4 = []
    samples5 = []
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    sample1 = model.decode(z_1)
    sample2 = model.decode(z_2)
    for alpha in alphas:
        out = alpha * (sample1) + (1 - alpha) * (sample2)
        samples1.append(out[0].cpu().data.numpy())
        samples2.append(out[1].cpu().data.numpy())
        samples3.append(out[2].cpu().data.numpy())
        samples4.append(out[3].cpu().data.numpy())
        samples5.append(out[4].cpu().data.numpy())

    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    samples3 = np.array(samples3)
    samples4 = np.array(samples4)
    samples5 = np.array(samples5)

    image_frame_dim = len(alphas)
    # pdb.set_trace()
    path = 'samples/vae/latent/interpol_1.png'
    print_samples = samples1[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_2.png'
    print_samples = samples2[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_3.png'
    print_samples = samples3[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_4.png'
    print_samples = samples4[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)
    
    path = 'samples/vae/latent/interpol_5.png'
    print_samples = samples5[:image_frame_dim * 1]
    print_samples = print_samples.transpose(0, 2, 3, 1)
    utils.save_images(print_samples, [1, image_frame_dim],path)

    return out


def z_eps_gan(z_sample):
    if not os.path.exists('samples/WGAN_GP_G/epsilon/'):
        os.makedirs('samples/WGAN_GP_G/epsilon/')
    epsilon = [2, 2.25, 2.5, 2.75, 3]
    for eps in epsilon:
        samples1 = []
        samples2 = []
        samples3 = []
        samples4 = []
        samples5 = []
        for dim in range(100):
            noise_copy = Variable(z_sample.clone(), requires_grad=False)
            # noise_copy[:,i] = noise_copy[:,i] + epsilon
            # pdb.set_trace()
            noise_copy = torch.transpose(noise_copy, 0, 1)
            noise_copy[dim] = noise_copy[dim] + eps
            noise_copy = torch.transpose(noise_copy, 0, 1)
            sample = model(noise_copy.cpu())
            samples1.append(sample[0].cpu().data.numpy())
            samples2.append(sample[1].cpu().data.numpy())
            samples3.append(sample[2].cpu().data.numpy())
            samples4.append(sample[3].cpu().data.numpy())
            samples5.append(sample[4].cpu().data.numpy())

        samples1 = np.array(samples1)
        samples2 = np.array(samples2)
        samples3 = np.array(samples3)
        samples4 = np.array(samples4)
        samples5 = np.array(samples5)

        count = 0
        max_diff = 0
        for cpt in range(100):
            for cpt2 in range(100):
                diff = np.max(abs(samples1[cpt]-samples1[cpt2]))
                # print('diff', diff)
                if diff != 0:
                    count += 1
                    if diff > max_diff:
                        max_diff = diff
        print('epsilon', eps, ' - number of img diff', count, ' - max_diff', max_diff)


        image_frame_dim = 10
        # pdb.set_trace()
        path = 'samples/WGAN_GP_G/epsilon/eps_' + str(eps) + '_1.png'
        print_samples = samples1[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/WGAN_GP_G/epsilon/eps_' + str(eps) + '_2.png'
        print_samples = samples2[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/WGAN_GP_G/epsilon/eps_' + str(eps) + '_3.png'
        print_samples = samples3[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/WGAN_GP_G/epsilon/eps_' + str(eps) + '_4.png'
        print_samples = samples4[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/WGAN_GP_G/epsilon/eps_' + str(eps) + '_5.png'
        print_samples = samples5[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)



def z_eps(z_sample):
    if not os.path.exists('samples/vae/epsilon/'):
        os.makedirs('samples/vae/epsilon/')
    epsilon = [120000000, 122500000, 125000000]
    for eps in epsilon:
        samples1 = []
        samples2 = []
        samples3 = []
        samples4 = []
        samples5 = []
        for dim in range(100):
            noise_copy = Variable(z_sample.clone(), requires_grad=False)
            # noise_copy[:,i] = noise_copy[:,i] + epsilon
            # pdb.set_trace()
            noise_copy = torch.transpose(noise_copy, 0, 1)
            noise_copy[dim] = noise_copy[dim] + eps
            noise_copy = torch.transpose(noise_copy, 0, 1)
            sample = model.decode(noise_copy)
            samples1.append(sample[0+15].cpu().data.numpy())
            samples2.append(sample[1+15].cpu().data.numpy())
            samples3.append(sample[2+15].cpu().data.numpy())
            samples4.append(sample[3+15].cpu().data.numpy())
            samples5.append(sample[4+15].cpu().data.numpy())

        samples1 = np.array(samples1)
        samples2 = np.array(samples2)
        samples3 = np.array(samples3)
        samples4 = np.array(samples4)
        samples5 = np.array(samples5)

        count = 0
        max_diff = 0
        for cpt in range(100):
            for cpt2 in range(100):
                diff = np.max(abs(samples1[cpt]-samples1[cpt2]))
                # print('diff', diff)
                if diff != 0:
                    count += 1
                    if diff > max_diff:
                        max_diff = diff
        print('epsilon', eps, ' - number of img diff', count, ' - max_diff', max_diff)


        image_frame_dim = 10
        # pdb.set_trace()
        path = 'samples/vae/epsilon/eps_' + str(eps) + '_1.png'
        print_samples = samples1[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/vae/epsilon/eps_' + str(eps) + '_2.png'
        print_samples = samples2[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/vae/epsilon/eps_' + str(eps) + '_3.png'
        print_samples = samples3[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/vae/epsilon/eps_' + str(eps) + '_4.png'
        print_samples = samples4[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)
        
        path = 'samples/vae/epsilon/eps_' + str(eps) + '_5.png'
        print_samples = samples5[:image_frame_dim * image_frame_dim]
        print_samples = print_samples.transpose(0, 2, 3, 1)
        utils.save_images(print_samples, [image_frame_dim, image_frame_dim],path)


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
    for i in range(16):
        z_latent = torch.randn((batch_size, z_dim)).to(device)
        visualize_all_sample(z_latent, model_name, i*64)

    print('print samples with epsilon')
    z_eps(z_latent)

    print('interpolate')
    z_latent = torch.randn((batch_size, z_dim)).to(device)
    z_latent2 = torch.randn((batch_size, z_dim)).to(device)
    z_interpolate(z_latent, z_latent2, 0)
    img_interpolate(z_latent, z_latent2, 0)

    
    print('WGAN_GP_G')
    model_name = 'WGAN_GP_G'
    input_size = 32
    model = generator(input_dim=z_dim, output_dim=3, input_size=input_size)

    load(model_name)
    model.eval()
    
    z_latent = torch.randn((batch_size, z_dim)).to(device)
    visualize_gan_sample(z_latent, model_name)
    print('print samples for #3 quantitative')
    for i in range(16):
        z_latent = torch.randn((batch_size, z_dim)).to(device)
        visualize_gan_all_sample(z_latent, model_name, i*64)

    print('print samples with epsilon')
    z_eps_gan(z_latent)


    print('interpolate')
    z_latent = torch.randn((batch_size, z_dim)).to(device)
    z_latent2 = torch.randn((batch_size, z_dim)).to(device)
    z_gan_interpolate(z_latent, z_latent2, 0)
    img_gan_interpolate(z_latent, z_latent2, 0)

