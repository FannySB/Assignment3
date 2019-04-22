import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
import numpy as np
from classify_svhn import Classifier
from scipy.linalg import sqrtm

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def get_train_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    trainset = torchvision.datasets.SVHN(
        SVHN_PATH, split='train',
        download=True,
        transform=classify_svhn.image_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
    )
    return trainloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator, testset_feature_iterator):
  
    feat_sample = np.asarray([s for s in sample_feature_iterator])
    mu_sample = np.mean(feat_sample, axis=0)
    sigma_sample = np.cov(feat_sample, rowvar=False)

    feat_test = np.asarray([s for s in testset_feature_iterator])
    mu_test = np.mean(feat_test, axis=0)
    sigma_test = np.cov(feat_test, rowvar=False)

    trace_mu_sample = np.trace(sigma_sample)
    trace_mu_test = np.trace(sigma_test)

    diff_mu = mu_sample - mu_test
    diff_mu2 = np.dot(diff_mu, diff_mu)

    offset = 0.001
    covmean, _ = sqrtm((sigma_sample + offset).dot(sigma_test + offset), disp=False)
    trace_covmean = np.trace(covmean)
    fid = diff_mu.dot(diff_mu) + trace_mu_sample + trace_mu_test - (2 * trace_covmean)

    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()   

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    # train_loader = get_train_loader(PROCESS_BATCH_SIZE)
    # train_f = extract_features(classifier, train_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
