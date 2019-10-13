import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
import imageio


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #Using notation from the theory 
        N = self.num_classes
        K = self.num_samples_per_class
        B = batch_size 

        #Instantiatie a consecutive vector of arbitrary labels to pass into their poorly coded funtion
        labels = np.arange(N)

        #Instatiate two zero matrices to index into and pass out 
        all_image_batches = np.zeros((B, K, N, self.dim_input))
        all_label_batches = np.zeros((B, K, N, N))

        #Iterating over batches, return all images and all labels for each batch
        #Building an image batch tensor w/ shape [B, K, N, 784] and a label batch tensor w/ shape [B, K, N, N]
        for b in range(B):

            #choose N random folders before calling image_labels
            N_folders_random = np.random.choice(folders, size=N, replace=False)

            #Takes a set of character folders and labels and returns paths to image files paired with labels.
            #N by K list of ["label", path]
            labels_imgs_list = get_images(N_folders_random, labels, K, False) 

            #Set label to unachievable value to trigger the first if below and begin counting samples per class
            n_prev = N + 1
            k = 0
            #Iterating over each label path tuple returned by their stupid function
            for n, image_path in labels_imgs_list:
                #Returns the sample number in K to k and sets n to label number in N
                n = int(n)
                if(n_prev != n):
                    k = 0
                    n_prev = n
                else:
                    k = k + 1
    
                #turns the currently all zeros final column into a one hot vector for batch_labels
                all_label_batches[b, k, n, n] = 1

                #Calls image_file_to_array to get flattened image vector for current path
                flattened_image = image_file_to_array(image_path, self.dim_input) 

                all_image_batches[b, k, n, :] = flattened_image

        #Take random permutations of order for each batch
        all_image_batches_unperm = np.array(all_image_batches)
        all_label_batches_unperm = np.array(all_label_batches)
        
        all_batches_perm = np.concatenate((all_image_batches_unperm, all_label_batches_unperm), axis=3)
        for b in range(B):
            np.random.shuffle(all_batches_perm[b,K-1,:,:])

        all_image_batches = all_batches_perm[:,:,:, 0:784]
        all_label_batches = all_batches_perm[:,:,:, 784:784+N]

        #############################

        return all_image_batches, all_label_batches
