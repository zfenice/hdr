# coding: utf-8

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# MTB implementation
def median_threshold_bitmap_alignment(img_list):
    median = [np.median(img) for img in img_list]
    binary_thres_img = [cv2.threshold(img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(len(img_list))]
    mask_img = [cv2.inRange(img_list[i], median[i]-20, median[i]+20) for i in range(len(img_list))]
 
    plt.imshow(mask_img[0], cmap='gray')
    plt.show()
    
    max_offset = np.max(img_list[0].shape)
    levels = 5

    global_offset = []
    for i in range(0, len(img_list)):
        offset = [[0,0]]
        for level in range(levels, -1, -1):
            scaled_img = cv2.resize(binary_thres_img[i], (0, 0), fx=1/(2**level), fy=1/(2**level))
            ground_img = cv2.resize(binary_thres_img[0], (0, 0), fx=1/(2**level), fy=1/(2**level))
            ground_mask = cv2.resize(mask_img[0], (0, 0), fx=1/(2**level), fy=1/(2**level))
            mask = cv2.resize(mask_img[i], (0, 0), fx=1/(2**level), fy=1/(2**level))
            
            level_offset = [0, 0]
            diff = float('Inf')
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    off = [offset[-1][0]*2+y, offset[-1][1]*2+x]
                    error = 0
                    for row in range(ground_img.shape[0]):
                        for col in range(ground_img.shape[1]):
                            if off[1]+col < 0 or off[0]+row < 0 or off[1]+col >= ground_img.shape[1] or off[0]+row >= ground_img.shape[1]:
                                continue
                            if ground_mask[row][col] == 255:
                                continue
                            error += 1 if ground_img[row][col] != scaled_img[y+off[0]][x+off[1]] else 0
                    if error < diff:
                        level_offset = off
                        diff = error
            offset += [level_offset]
        global_offset += [offset[-1]]
    return global_offset


def hdr_debvec(img_list, exposure_times, number_of_samples_per_dimension=20):    
    B = np.log(exposure_times) #get the logarithm of exposures
    l = 100      # l is lambda, the constant that determines the amount of smoothness
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]  #create the weighted function to emphasize the middle points

    #samples = []
    width = img_list[0].shape[0]
    height = img_list[0].shape[1]
    width_step = math.floor(width / number_of_samples_per_dimension)
    height_step = math.floor(height / number_of_samples_per_dimension)

    x = 0
    y = 0

    Z = np.zeros((len(img_list), number_of_samples_per_dimension*number_of_samples_per_dimension))
    for img_index, img in enumerate(img_list):
        y = 0
        for i in range(number_of_samples_per_dimension):
            x = 0
            for j in range(number_of_samples_per_dimension):
                if x < width and y < height:
                    pixel = img[x, y]
                    Z[img_index, i * number_of_samples_per_dimension + j] = pixel
                x += width_step
            y += height_step
    
    return response_curve_solver(Z, B, l, w)


# Implementation of paper's Equation(3) with weight
# Z pixel samples
# B log of exposures
# l lamda smoothness coefficient
# w smoothness weight
def response_curve_solver(Z, B, l, w):
    n = 256
    images_num = Z.shape[0]
    pixels_num = Z.shape[1] # samples per image
    A = np.zeros(shape=(images_num * pixels_num + (n+1), n + pixels_num), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0)), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(pixels_num):
        for j in range(images_num):
            z = int(Z[j][i])
            wij = w[z]
            A[k][z] = wij       #coefficient of g(x) where x is between (0, 256)
            A[k][n+i] = -wij    #coefficient of lE(x) where x is (0, pixels_num)
            b[k] = wij*B[j]
            k += 1
    
    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


# In[5]:

# Implementation of paper's Equation(6)
def construct_radiance_map(g, Z, ln_t, w):
    images_num = len(Z)
    pixels_num = len(Z[0])
    acc_E = [0]*pixels_num
    ln_E = [0]*pixels_num
    
    for i in range(pixels_num):
        acc_w = 0
        for j in range(images_num):
            z = Z[j][i]
            acc_E[i] += w[z]*(g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else acc_E[i]
    
    return ln_E

def construct_hdr(img_list, response_curve, exposure_times):
    # Construct radiance map for each channels
    img_size = img_list[0][0].shape
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]
    ln_t = np.log(exposure_times)
    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

    # construct radiance map for BGR channels
    for i in range(3):
        print(' - Constructing radiance map for {0} channel .... '.format('BGR'[i]), end='', flush=True)
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(np.exp(E), img_size)
        print('done')

    return hdr

#not using opencv
# def tonemap(hdr):
#     gamma = 2
#     tonemap = cv2.createTonemapDrago(gamma)
#     ldr = tonemap.process(hdr)
#     return ldr * 255

def tonemap(hdr):
    #amplify low irradiance more than high irradiance
    gamma = 1.3
    r1 = 1 - np.exp(-hdr * gamma)     #low irradiance
    r2 = 0.6 + 0.07 * hdr             #high irradiance
    ldr = np.minimum(r1, r2)
    return np.clip(ldr * 255, 0, 255).astype('uint8')

# main
if __name__ == '__main__': 
    img_dir = 'church'
    output_hdr_filename = 'output.png'

    # Loading exposure images into a list
    print('Reading input images.... ', end='')
    exposure_times = [32,16,8,4,2,1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625]
    img_paths = ['church01.png', 'church02.png', 'church03.png','church04.png','church05.png','church06.png','church07.png','church08.png','church09.png','church10.png','church11.png','church12.png','church13.png','church14.png','church15.png','church16.png']
    images = [cv2.imread(os.path.join(img_dir, path)) for path in img_paths]
    image_channel = [cv2.split(image) for image in images]
    img_list_b = [x[0] for x in image_channel]
    img_list_g = [x[1] for x in image_channel]
    img_list_r = [x[2] for x in image_channel]
    print('done')

    # Solving response curves
    print('Solving response curves .... ', end='')
    gb, _ = hdr_debvec(img_list_b, exposure_times)
    gg, _ = hdr_debvec(img_list_g, exposure_times)
    gr, _ = hdr_debvec(img_list_r, exposure_times)
    print('done')

    # Show response curve
    print('Saving response curves plot .... ', end='')
    plt.figure(figsize=(10, 10))
    plt.plot(gr, range(256), 'rx')
    plt.plot(gg, range(256), 'gx')
    plt.plot(gb, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig('response-curve.png')
    print('done')

    print('Constructing HDR image: ')
    hdr = construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)
    print('done')

    # Display Radiance map with pseudo-color image (log value)
    print('Saving pseudo-color radiance map .... ', end='')
    plt.figure(figsize=(12,8))
    plt.imshow(np.log(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('radiance-map.png')
    print('done')

    print('Saving HDR image .... ', end='')
    cv2.imwrite('hdr.png', hdr)
    print('done')

    #save tonemap graph
    print('Tonemapping.... ', end='')
    x = np.arange(0, 6, 0.1)
    y = tonemap(x)
    plt.figure(figsize=(10, 10))
    plt.plot(x, y)
    plt.savefig('tonemapping.png')
    print('done')

    print('Saving LDR image .... ', end='')
    ldr = tonemap(hdr)
    cv2.imwrite('ldr.png', ldr)
    print('done')
