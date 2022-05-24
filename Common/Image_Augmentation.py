import numpy as np

#Reinforcement Learning with Augmented Data, Laskin et al, 2020
def center_crop(images, output_size, data_format='channels_first'):
    '''
    :param images: numpy array
    :param output_size: int or tuple value for output image size
    :param data_format: 'channels_first', or 'channels_last'
    :return:
    '''
    assert images.ndim in (3, 4), "Image type must be numpy array, and its dimension must be 3 or 4"
    original_ndim = images.ndim
    if original_ndim == 3:
        images = np.expand_dims(images, axis=0)

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == 'channels_first':
        b, c, h, w = images.shape
    else:
        b, h, w, c = images.shape

    assert h >= output_size[0] and w >= output_size[1]

    new_h, new_w = output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    if data_format == 'channels_first':
        images = images[:, :, top:top + new_h, left:left + new_w]
    else:

        images = images[:, top: top + new_h, left:left + new_w, :]

    if original_ndim == 3:
        images = images[0]

    return images

#Reinforcement Learning with Augmented Data, Laskin et al, 2020
def random_crop(images, output_size, data_format='channels_first'):
    assert images.ndim in (3, 4), "Image type must be numpy array, and its dimension must be 3 or 4"
    original_ndim = images.ndim

    if original_ndim == 3:
        images = np.expand_dims(images, axis=0)

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == 'channels_first':
        b, c, h, w = images.shape
    else:
        b, h, w, c = images.shape

    assert h >= output_size[0] and w >= output_size[1]

    crop_max_h = h - output_size[0] + 1
    crop_max_w = w - output_size[1] + 1

    w1 = np.random.randint(0, crop_max_w, b)
    h1 = np.random.randint(0, crop_max_h, b)

    if data_format == 'channels_first':
        cropped = np.empty((images.shape[0], images.shape[1], output_size[0], output_size[1]), dtype=images.dtype)
        for i, (images, w11, h11) in enumerate(zip(images, w1, h1)):
            cropped[i] = images[:, h11:h11 + output_size[0], w11: w11 + output_size[1]]

    else:
        cropped = np.empty((images.shape[0], output_size[0], output_size[1], images.shape[-1]), dtype=images.dtype)
        for i, (images, w11, h11) in enumerate(zip(images, w1, h1)):
            cropped[i] = images[h11:h11 + output_size[0], w11: w11 + output_size[1], :]

    if original_ndim == 3:
        cropped = cropped[0]
    return cropped

#Reinforcement Learning with Augmented Data, Laskin et al, 2020
def center_translate(images, output_size, data_format='channels_first'):
    assert images.ndim in (3, 4), "Image type must be numpy array, and its dimension must be 3 or 4"
    original_ndim = images.ndim

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    if data_format == 'channels_first':
        b, c, h, w = images.shape
    else:
        b, h, w, c = images.shape

    assert output_size[0] >= h  and output_size[1] >= w


    if data_format == 'channels_first':
        outputs = np.zeros((b, c, output_size[0], output_size[1]), dtype=images.dtype)
    else:
        outputs = np.zeros((b, output_size[0], output_size[1], c), dtype=images.dtype)

    h1 = (output_size[0] - h)//2
    w1 = (output_size[1] - w)//2

    for output, image in zip(outputs, images):
        if data_format == 'channels_first':
            output[:, h1: h1+h, w1: w1+w] = image
        else:
            output[h1: h1+h, w1: w1+w, :] = image

    if original_ndim == 3:
        outputs = outputs[0]

    return outputs


#Reinforcement Learning with Augmented Data, Laskin et al, 2020
def random_translate(images, output_size, return_random_idxs=False, h1s=None, w1s=None, data_format='channels_first'):
    assert images.ndim in (3, 4), "Image type must be numpy array, and its dimension must be 3 or 4"
    original_ndim = images.ndim

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    if data_format == 'channels_first':
        b, c, h, w = images.shape
    else:
        b, h, w, c = images.shape

    assert output_size[0] >= h and output_size[1] >= w


    if data_format == 'channels_first':
        outputs = np.zeros((b, c, output_size[0], output_size[1]), dtype=images.dtype)
    else:
        outputs = np.zeros((b, output_size[0], output_size[1], c), dtype=images.dtype)

    h1s = np.random.randint(0, output_size[0] - h + 1, b) if h1s is None else h1s
    w1s = np.random.randint(0, output_size[1] - w + 1, b) if w1s is None else w1s

    for output, image, h1, w1 in zip(outputs, images, h1s, w1s):
        if data_format == 'channels_first':
            output[:, h1: h1+h, w1: w1+w] = image

        else:
            output[h1: h1+h, w1: w1+w, :] = image

    if original_ndim == 3:
        outputs = outputs[0]

    if return_random_idxs:
        return outputs, dict(h1s=h1s, w1s=h1s)

    return outputs

#Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels, Kostrikov et al, 2021
#nn.ReplicationPad2d
def random_shift(images, image_pad=4, data_format='channels_first'):
    assert images.ndim in (3, 4), "Image type must be numpy array, and its dimension must be 3 or 4"
    original_ndim = images.ndim

    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    if data_format == 'channels_first':
        #b, c, h, w = images.shape
        pad_width = ((0, 0), (0,0), (image_pad, image_pad), (image_pad, image_pad))

    else:
        #b, h, w, c = images.shape
        pad_width = ((0, 0), (image_pad, image_pad), (image_pad, image_pad), (0, 0))

    outputs = np.pad(images, pad_width, mode='edge')
    if original_ndim == 3:
        outputs = outputs[0]

    return outputs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = np.ones((100, 100, 3))
    print(random_shift(test, data_format='channels_last').shape)
    exit()
    plt.imshow(test)
    plt.show()
    print(random_crop(test, 80, data_format='channels_last').shape)
    print(random_translate(test, 108, False, data_format='channels_last').shape)
    plt.imshow(center_translate(test, 108, data_format='channels_last'))
    plt.show()
    print(np.unique(random_translate(test, 108, False, data_format='channels_last')))
    #print(random_crop_image(test, (50, 64), data_format='channels_first').shape)