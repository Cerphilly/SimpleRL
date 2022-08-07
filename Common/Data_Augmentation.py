import numpy as np
import tensorflow as tf
import numbers

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

def grayscale(images, p=0.3):#same as RAD's grayscale, but output dtype is uint8 and numpy array
    assert images.dtype == 'uint8'

    b, c, h, w = images.shape
    frames = c // 3
    gray_images = np.reshape(images.astype(np.float32), [b, frames, 3, h, w])
    gray_images = gray_images[:,:,0,:,:] * 0.2989 + gray_images[:,:,1,:,:] * 0.587 + gray_images[:,:,2,:,:] * 0.114
    gray_images = gray_images.astype(np.uint8).astype(np.float32)
    gray_images = gray_images[:,:,None,:,:] * np.ones([1,1,3,1,1], dtype=np.float32)

    mask = np.random.uniform(0, 1, size=(b,)) <= p

    images = np.reshape(images, gray_images.shape)
    mask = (mask[:, None] * np.ones([1, frames]))
    mask = mask[:, :, None, None, None]
    output = mask * gray_images + (1 - mask) * images

    output = np.reshape(output, [b, -1, h, w])

    return output.astype(np.uint8)

def cutout(images, min_cut=10, max_cut=30):
    b, c, h, w = images.shape
    w1 = np.random.randint(min_cut, max_cut, b)
    h1 = np.random.randint(min_cut, max_cut, b)

    new_images = np.empty((b, c, h, w), dtype=images.dtype)
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cut_img = img.copy()
        cut_img[:,h11:h11+h11, w11:w11+w11] = 0
        new_images[i] = cut_img

    return new_images



def cutout_color(images, min_cut=10, max_cut=30):
    b, c, h, w = images.shape
    w1 = np.random.randint(min_cut, max_cut, b)
    h1 = np.random.randint(min_cut, max_cut, b)

    new_images = np.empty((b, c, h, w), dtype=images.dtype)
    rand_box = np.random.randint(0, 255, size = (b, c))
    for i, (img, w11, h11) in enumerate(zip(images, w1, h1)):
        cut_img = img.copy()
        cut_img[:,h11: h11+h11, w11:w11+w11] = np.tile(rand_box[i].reshape(-1, 1, 1), (1,) + cut_img[:, h11:h11+h11, w11:w11+w11].shape[1:])

        new_images[i] = cut_img

    return new_images



def flip(images, p=0.2):
    b, c, h, w = images.shape
    flipped_images = np.flip(images, axis=3)
    mask = np.random.uniform(0, 1, size=(b,)) <= p
    frames = images.shape[1]

    images = np.reshape(images, flipped_images.shape)

    mask = mask[:, None] * np.ones([1, frames]).astype(np.bool)
    mask = mask[:, :, None, None]
    output = mask * flipped_images + (1 - mask) * images

    output = np.reshape(output, [b, -1, h, w])
    output = output.astype(np.uint8)

    return output


def rotation(images, p=1.):#input: np.array, output: np.array
    b, c, h, w = images.shape

    rot90_images = np.rot90(images, k=1, axes=[2, 3])
    rot180_images = np.rot90(images, k=2, axes=[2, 3])
    rot270_images = np.rot90(images, k=3, axes=[2, 3])

    mask = (np.random.uniform(0., 1., size=(b,)) <= p).astype('int32')

    mask *= np.random.randint(1, 4, size=(b,))
    masks = [np.zeros_like(mask) for _ in range(4)]
    for i, m in enumerate(masks):
        m[np.where(mask == i)] = 1
        m = m[:, None] * np.ones([1, c], dtype=np.uint8)
        m = m[:, :, None, None]
        masks[i] = m

    output = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    output = np.reshape(output, [b, -1, h, w])
    output = output.astype(np.uint8)

    return output


def convolution(images):
    #Network randomization: A simple technique for generalization in deep reinforcement learning, Lee et al, 2020
    #augments the image color by passing the input observation through a random convolutional layer

    num_batch, num_stack_channel, img_h, img_w = images.shape
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)
    images = images / 255.0

    rand_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, data_format='channels_first', use_bias=False, padding='same', kernel_initializer='glorot_normal')

    for trans_index in range(num_trans):

        temp_imgs = images[trans_index * batch_size:(trans_index + 1) * batch_size].astype(np.float32)
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w)  # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = tf.concat((total_out, rand_out), 0)

    total_out = tf.reshape(total_out, (-1, num_stack_channel, img_h, img_w)) * 255.0

    return (total_out.numpy()).astype(np.uint8)



def rgb2hsv(rgb, eps=1e-8):
    # Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    Cmax = np.max(rgb, axis=1)
    Cmin = np.min(rgb, axis=1)

    delta = Cmax - Cmin

    hue = np.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3]))
    hue[Cmax == r] = (((g - b)/(delta + eps)) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r)/(delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g)/(delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0

    hue = hue / 6.
    hue = np.expand_dims(hue, axis=1)

    saturation = (delta) / (Cmax + eps)
    saturation[Cmax == 0.] = 0.
    saturation = np.expand_dims(saturation, axis=1)

    value = Cmax
    value = np.expand_dims(value, axis=1)

    return np.concatenate((hue, saturation, value), axis=1)

def hsv2rgb(hsv):
    # Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287
    hsv = np.clip(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]
    c = value * saturation
    x = -c * (np.abs((hue / 60.) % 2 -1) - 1)
    m = np.expand_dims((value - c), axis=1)

    rgb_prime = np.zeros_like(hsv)

    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]

    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]

    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]

    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]

    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]

    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]

    rgb = rgb_prime + np.concatenate((m, m, m), axis=1)

    return np.clip(rgb, 0, 1)


class random_color_jitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5, p=1, batch_size=128, stack_size=3):
        self.brightness = self.check_input(brightness, 'brightness')
        self.contrast = self.check_input(contrast, 'contrast')
        self.saturation = self.check_input(saturation, 'saturation')
        self.hue = self.check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.prob = p
        self.batch_size = batch_size
        self.stack_size = stack_size

    def check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        factor = np.random.uniform(*self.contrast, size=self.batch_size)
        factor = np.reshape(factor, (-1, 1))
        factor = np.repeat(factor, self.stack_size, axis=1)
        factor = np.reshape(factor, -1)
        means = np.mean(x, axis=(2,3), keepdims=True)

        return np.clip((x-means) * np.reshape(factor, (len(x), 1, 1, 1)), 0, 1)

    def adjust_hue(self, x):
        factor = np.random.uniform(*self.hue, size=self.batch_size)
        factor = np.reshape(factor, (-1, 1))
        factor = np.repeat(factor, self.stack_size, axis=1)
        factor = np.reshape(factor, -1)
        h = x[:, 0, :, :]
        h += np.reshape(factor * (255./360.), newshape=(len(x), 1, 1))
        h = (h % 1)
        x[:, 0, :, :] = h

        return x

    def adjust_brightness(self, x):
        factor = np.random.uniform(*self.brightness, size=self.batch_size)
        factor = np.reshape(factor, (-1, 1))
        factor = np.repeat(factor, self.stack_size, axis=1)
        factor = np.reshape(factor, -1)

        x[:, 2, :, :] = np.clip(x[:, 2, :, :] * np.reshape(factor, (len(x), 1, 1)), 0, 1)

        return np.clip(x, 0, 1)

    def adjust_saturate(self, x):
        factor = np.random.uniform(*self.saturation, size=self.batch_size)
        factor = np.reshape(factor, (-1, 1))
        factor = np.repeat(factor, self.stack_size, axis=1)
        factor = np.reshape(factor, -1)

        x[:,1,:,:] = np.clip(x[:, 1, :, :] * np.reshape(factor, (len(x), 1, 1)), 0, 1)

        return np.clip(x, 0, 1)

    def transform(self, inputs):
        hsv_transform_list = [rgb2hsv, self.adjust_brightness, self.adjust_hue, self.adjust_saturate, hsv2rgb]
        rgb_transform_list = [self.adjust_contrast]

        if np.random.uniform() >= 0.5:
            transform_list = rgb_transform_list + hsv_transform_list
        else:
            transform_list = hsv_transform_list + rgb_transform_list

        for t in transform_list:
            inputs = t(inputs)

        return inputs


    def __call__(self, inputs):
        b, c, h, w = inputs.shape
        inputs = inputs / 255.0
        inputs = np.reshape(inputs, (-1, 3, h, w))

        random_inds = np.random.choice([True, False], len(inputs), p=[self.prob, 1 - self.prob])
        if random_inds.sum() > 0:
            inputs[random_inds] = self.transform(inputs[random_inds])

        inputs = np.reshape(inputs, (b, c, h, w))
        return (inputs * 255.0).astype(np.uint8)

def no_aug(x):
    return x

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
    import dmc2gym, cv2
    import numpy as np
    from Common.Utils import FrameStack
    import time
    from tabulate import tabulate
    import tensorflow as tf
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    def now():
        return time.time()

    def secs(t):
        s = now() - t
        tot = round((1e5 * s) / 60, 1)
        return round(s, 3), tot

    img = np.load('../data_sample.npy', allow_pickle=True)

    img = np.concatenate([img, img, img], 1)
    print(img.shape)
    cv2.imshow("img", np.transpose(img[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)

    #random crop
    t = now()
    img1 = random_crop(img, 64)
    s1, tot1 = secs(t)
    cv2.imshow("img1", np.transpose(img1[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    #random grayscale
    t = now()
    img2 = grayscale(img)
    s2, tot2 = secs(t)
    cv2.imshow("img2", np.transpose(img2[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    #normal cutout
    t = now()
    img3 = cutout(img,10,30)
    s3,tot3 = secs(t)
    cv2.imshow("img3", np.transpose(img3[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    #color cutout
    t = now()
    img4 = cutout_color(img,10,30)
    s4,tot4 = secs(t)
    cv2.imshow("img4", np.transpose(img4[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    #flip
    t = now()
    img5 = flip(img)
    s5,tot5 = secs(t)
    cv2.imshow("img5", np.transpose(img5[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    # rotate
    t = now()
    img6 = rotation(img,p=1)
    s6,tot6 = secs(t)
    cv2.imshow("img6", np.transpose(img6[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    # rand conv
    t = now()
    img7 = convolution(img)
    s7,tot7 = secs(t)
    cv2.imshow("img7", np.transpose(img7[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    #color_jitter
    transform = random_color_jitter()
    t = now()

    img8 = transform(img)
    s8,tot8 = secs(t)
    cv2.imshow("img8", np.transpose(img8[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)
    t = now()
    img9 = translate(img, 100)
    s9, tot9 = secs(t)
    cv2.imshow("img9", np.transpose(img9[0][0:3, :, :], (1, 2, 0)))
    cv2.waitKey(0)

    print(tabulate([['Crop', s1, tot1],
                    ['Grayscale', s2, tot2],
                    ['Normal Cutout', s3, tot3],
                    ['Color Cutout', s4, tot4],
                    ['Flip', s5, tot5],
                    ['Rotate', s6, tot6],
                    ['Rand Conv', s7, tot7],
                   ['Color jitter', s8, tot8],
                   ['Translate'], s9, tot9],
                   headers=['Data Aug', 'Time / batch (secs)', 'Time / 100k steps (mins)']))
