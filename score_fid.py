import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
import numpy as np
from classify_svhn import Classifier

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


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    import pdb; pdb.set_trace()

    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):

    len_iterator = len(list(testset_feature_iterator))
    mu_sample = np.zeros(len_iterator)
    sigma_sample = np.zeros(len_iterator)
    trace_sample = 1
    counter = 0
    for feat_sample in sample_feature_iterator:
        # import pdb; pdb.set_trace()
        mu_sample[counter] = np.mean(feat_sample)
        sigma_sample[counter] = np.var(feat_sample)
        trace_sample = trace_sample * sigma_sample
        counter += 1

    mu_test = np.zeros(len_iterator)
    sigma_test = np.zeros(len_iterator)
    trace_test = 1
    counter = 0
    for feat_test in testset_feature_iterator:
        mu_test[counter] = np.mean(feat_test)
        sigma_test[counter] = np.var(feat_test)
        trace_test = trace_test * sigma_test
        counter += 1

    import pdb; pdb.set_trace()
    covar = 2*(np.sqrt(sigma_sample.dot(sigma_test)))
    trace_covar = 1
    counter = 0
    for x in covar:
        trace_covar = trace_covar * x
        counter += 1

    mu_diff = mu_sample - mu_test

    return mu_diff**2 + trace_sample + trace_test - trace_covar


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

    import pdb; pdb.set_trace()
    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
