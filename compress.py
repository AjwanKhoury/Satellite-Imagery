import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

def compress_img(image_name):
    compress_process = []
    # load the image to memory
    img = Image.open(image_name)
    # print the original image shape
    compress_process.append("[*] Image shape:{}".format(img.size))
    # get the original image size in bytes
    image_size = os.path.getsize(image_name)
    if image_size > 20*1024:
        quality = int(10*1024/image_size * 100)
    else:
        quality = 100
    
    # print the size before compression/resizing
    compress_process.append("[*] Size before compression:{}".format(get_size_format(image_size)))
    compress_process.append("[*] New Image shape:{}".format(img.size))
        
    # make new filename appending _compressed to the original file name
    dirname, filname = os.path.split(image_name)
    # change the extension to JPEG
    new_filename = 'test/Compress/Compressed/{}'.format(filname)

    try:
        # save the image with the corresponding quality and optimize set to True
        img.save(new_filename, quality=quality, optimize=True)
    except OSError:
        # convert the image to RGB mode first
        img = img.convert("RGB")
        # save the image with the corresponding quality and optimize set to True
        img.save(new_filename, quality=quality, optimize=True)
        
    compress_process.append("[*] New file saved:{}".format(new_filename))
    # get the new image size in bytes
    new_image_size = os.path.getsize(new_filename)
    # print the new size in a good format
    compress_process.append("[*] Size after compression:{}".format(get_size_format(new_image_size)))
    # calculate the saving bytes
    saving_diff = new_image_size - image_size
    # print the saving percentage
    compress_process.append(f"Image size change: {saving_diff/image_size*100:.2f}% of the original image size.")
    
    org_img = img_to_array(load_img(image_name))
    compress_img = img_to_array(load_img(new_filename))
    
    w, h, ch = compress_img.shape
    pair_img = np.zeros([w, 2*h, ch])
    pair_img[:, :h, :] = org_img
    pair_img[:, h:, :] = compress_img
    pair_img = np.require(pair_img, np.uint8, 'C')
    
    return compress_process, pair_img