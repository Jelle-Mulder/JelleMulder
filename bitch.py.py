import cv2 as cv2
import numpy as np
from tqdm import tqdm

img = cv2.imread('bleskive.jpg')

H, W, _ = img.shape
img_radius = min(W, H)/2

outer_radius = 2
inner_radius = 0.5

phi = np.pi/4
mu = np.tan(phi)
revert_matrix = np.array([[1, mu], [-mu, 1]])

width = np.log(outer_radius) - np.log(inner_radius)

def transform_coords(x,y):
    return ((x-W/2)*outer_radius/img_radius, (H/2 - y) * outer_radius/img_radius)

def revert_coords(x,y):
    return (x/outer_radius*img_radius+W/2, H/2 - y*img_radius/outer_radius)

def transform_output_coords(x,y):
    return ((x-resolution[0]/2)*output_radius/output_radius_img, (resolution[1]/2-y)*output_radius / output_radius_img)


def transform_image():
    trans_img = np.zeros((W, H, 2))

    for x in range(W):
        for y in range(H):
            X, Y = transform_coords(x, y)
            sqr_dist = X*X + Y*Y
            if sqr_dist > outer_radius*outer_radius or sqr_dist < inner_radius * inner_radius:
                trans_img[x, y] = [0, -1]
            
            angle = np.angle(X+1j*Y)
            angle += 2*np.pi if angle < 0 else 0
            trans_img[x,y] = [np.log(np.sqrt(sqr_dist)), angle]
    
    trans_img[:,:,0] -= np.log(inner_radius)
    return trans_img


def transform_output_image():
    trans_output = np.zeros((*resolution, 2))
    for y in range(resolution[0]):
        for x in range(resolution[1]):
            X, Y = transform_output_coords(x,y)
            trans_output[y, x] = [X,Y]


    dists = np.log(np.sqrt(trans_output[:, :, 0]**2 + trans_output[:, :, 1]**2))
    angles = np.angle(trans_output[:, :, 0] + 1j * trans_output[:, :, 1])
    angles[angles < 0] += 2*np.pi
    angles += mu*dists


    trans_output[:,:, 0] = dists
    trans_output[:,:, 1] = angles
    for y in range(resolution[0]):
        for x in range(resolution[1]):
            trans_output[y,x] = np.dot(revert_matrix, trans_output[y,x])
    
    trans_output[:, :, 0] -= np.log(inner_radius)
    trans_output[:, :, 0] %= width
    trans_output[:, :, 0] += np.log(inner_radius)
    return trans_output

def get_pixel(x,y):
    Z = trans_img - trans_output[x,y]
    dists = Z[:, :, 0] ** 2 + Z[:,:, 1]**2
    M = np.argmax(-dists)
    Y = M // H
    X = M % H
    return img[X, Y]

def get_pixels():
    mags = np.exp(trans_output[:, :, 0])
    angles = trans_output[:,:,1]
    X = mags * np.cos(angles)
    Y = mags * np.sin(angles)
    x, y = revert_coords(X, Y)

    return x, y

    

resolution = (1000,1000)
output_radius = 2
output_radius_img = min(*resolution)/2
output_img = np.zeros((*resolution, 3), np.uint8)

trans_output = transform_output_image()




X, Y = get_pixels()


for y in tqdm(range(resolution[0])):
    for x in range(resolution[1]):
        if x == resolution[0]/2 and y == resolution[1]/2:
            output_img[y,x] = img[H//2, W//2]
            continue

        output_img[y, x] = img[int(Y[y,x]), int(X[y,x])]

cv2.imwrite('output.jpg', output_img)


