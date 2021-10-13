
from matplotlib.pyplot import get_current_fig_manager
import skimage.io as skio
import skimage.draw as draw
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import subprocess
import cv2

from scipy.spatial import Delaunay
from os import listdir
from os.path import isfile, join

#zion_left = skio.imread("img/zion_left.jpg")
#zion_right = skio.imread("img/zion_right.jpg")

def show(img):
    skio.imshow(img)
    skio.show()

def save(img, name):
    skio.imsave(name, img)

def jpg_name(name):
    return "img/" + name + ".jpg"

def png_name(name):
    return "img/" + name + ".png"

def population_name(name):
    return "img/population/" + name + ".jpg"

def read_png(name, astype = "float64"):
    png = skio.imread(png_name(name))
    if len(png.shape) == 2:
        # Handle the grayscale case
        return png.astype(astype)
    # Remove the alpha channel.
    return png[:,:,:3].astype(astype)

def read_jpg(name, astype = "float64"):
    jpg = skio.imread(jpg_name(name))
    if len(jpg.shape) == 2:
        # Handle the grayscale case
        return jpg.astype(astype)
    return jpg[:,:,:3].astype(astype)
    
def points_name(name):
    return "points/" + name + ".points"

def select_points(image, num_points, name):
    """ Choose NUM_POINTS on IMAGE and save them as NAME. """
    
    points = []
    plt.imshow(image)
    points = plt.ginput(num_points, 0)
    # points = retrieve_points(name)
    # points = points[:-4]

    plt.close()
    pickle_name = re.split("\.", name)[0] + ".points"
    pickle.dump(points, open("points/" + pickle_name, "wb"))
    return points

def retrieve_points(name):
    """ Retrieve points for NAME from the points directory. """
    points = pickle.load(open(points_name(name), "rb"))
    return np.array(points)

def retrieve_points_path(path):
    """ Retrieve points for NAME from the points directory. """
    points = pickle.load(open(path, "rb"))
    return np.array(points)

def display_points(imageA_name, imageB_name):
    """ Display selected points over two images. """
    pointsA = retrieve_points(imageA_name)
    pointsB = retrieve_points(imageB_name)
    point_display_name_A = jpg_name(re.split("\.", imageA_name)[-1] + "points")
    point_display_name_B = jpg_name(re.split("\.", imageB_name)[-1] + "points")

    imgA = skio.imread(jpg_name(imageA_name))
    plt.imshow(imgA)
    for i, point in enumerate(pointsA):
        plt.scatter(point[0], point[1], color="red")
        plt.annotate(str(i), (point[0], point[1]))
    plt.savefig(point_display_name_A)
    plt.close()

    imgB = skio.imread(jpg_name(imageB_name))
    plt.imshow(imgB)
    for i, point in enumerate(pointsB):
        plt.scatter(point[0] - 1, point[1] - 1, color="pink")
        plt.annotate(str(i), (point[0] + 4, point[1] + 4))
    plt.savefig(point_display_name_B)
    plt.close()

    point_display_A = skio.imread(point_display_name_A)
    point_display_B = skio.imread(point_display_name_B)
    skio.imshow(np.concatenate([point_display_A, point_display_B]))
    skio.show()


def computeH_SVD(im1_points, im2_points):
    """

    Unused.


        Input: im1_points, im2_points: matrix of the form
            [
                [ X  Y ]
                [ X  Y ]
                [ X  Y ]
                [ X  Y ]
                            ]

        Returns H, 3x3 homography matrix of form
            [
                [ a b c ]
                [ d e f ]
                [ g h 1 ]
                            ]

        Additional credit goes to online discussion on StackOverflow & Reddit,
        where I found some diagrams & code like the following:
        https://math.stackexchange.com/questions/494238/how-to-compute
        -homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    """

    #im1_points = [[i, i + 1] for i in range(0, 18, 2)]
    #im2_points = [[i, i + 1] for i in range(0, 18, 2)]
    
    A = []
    for i in range(0, len(im1_points)):
        x, y = im1_points[i][1], im1_points[i][0]
        u, v = im2_points[i][1], im2_points[i][0]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H


def a_matrix_rows(p1, p2):
    top = np.array([p2[0], p2[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[1]*p1[0]])
    bottom = np.array([0, 0, 0, p2[0], p2[1], 1, -p2[0]*p1[1], -p2[1]*p1[1]])
    return np.vstack([top, bottom])

def computeH(points1, points2):
    """ 
    Use least squares to compute the homography matrix from the set of
    POINTS1 to POINTS2.
    """

    # Stack all of the points for IMAGE1 on top of each other.
    b = np.zeros((len(points1)*2, 1))
    i = 0
    for i in range(len(points1)):
        b[i*2] = points1[i][0]
        b[i*2+1] = points1[i][1]
    
    # Stack the other side of the equation on top of each other.
    A = a_matrix_rows(points1[0], points2[0])
    for i in range(1, len(points1)):
        newline = a_matrix_rows(points1[i], points2[i])
        A = np.vstack([A, newline])
    H_arr = (np.linalg.lstsq(A, b, rcond=-1)[0]).T[0]
    H = np.matrix([[H_arr[0], H_arr[1], H_arr[2]],
                   [H_arr[3], H_arr[4], H_arr[5]],
                   [H_arr[6], H_arr[7], 1.     ]])
    return H

def warpImage(image, image2, H):
    """ Using IMAGE as a baseline, stretch IMAGE2 into it via homography matrix H. """
    
    # Determine the corners of the image after warping it.
    shape = image2.shape
    max_x = shape[1]
    max_y = shape[0]

    bottom_left = np.matrix([[0], [max_y], [1]])
    bottom_right = np.matrix([[max_x], [max_y], [1]])
    top_left = np.matrix([[0], [0], [1]])
    top_right = np.matrix([[max_x], [0], [1]])

    corners = [bottom_left, bottom_right,
                       top_left, top_right]

    corners = [H @ point for point in corners]

    # Normalize.
    corners = [point / point[2] for point in corners]

    # The image may be stretched far left of its original left border (0)
    # and far to the right of its original right border (image.shape[1])
    # Alternatively, it could actually be smaller than the original
    # dimensions, so we'll keep the original dimensions if possible.
    transformed_x_max = max(corners, key=lambda x: x[0])[0].astype(np.int)
    transformed_y_max = max(corners, key=lambda x: x[1])[1].astype(np.int)
    transformed_x_min = min(corners, key=lambda x: x[0])[0].astype(np.int)
    transformed_y_min = min(corners, key=lambda x: x[1])[1].astype(np.int)

    right_edge, bottom_edge, left_edge, top_edge = \
        [transformed_x_max[0, 0], transformed_y_max[0, 0], transformed_x_min[0, 0], transformed_y_min[0, 0]]
    right_edge = max(right_edge, image.shape[1], image2.shape[1])
    bottom_edge = max(bottom_edge, image.shape[0], image2.shape[0])

    # https://scikit-image.org/docs/0.14.x/api/skimage.draw.html#skimage.draw.polygon
    # Draw a polygon that surrounds the borders of the warped image2.
    mask = draw.polygon([0, right_edge + abs(left_edge), right_edge + abs(left_edge), 0],
                        [0, 0, bottom_edge + abs(top_edge), bottom_edge + abs(top_edge)])
    
    mask = np.matrix(np.vstack([mask, np.ones(len(mask[0]))]))

    # Reverse the polygon such that the points inside it map to the original image2.
    mask2 = np.linalg.inv(H) @ mask
    cc, rr, w = mask2

    # Make these into arrays of indices rather than matrices.
    cc = np.squeeze(np.asarray(cc))
    rr = np.squeeze(np.asarray(rr))
    w = np.squeeze(np.asarray(w))
    cc = (cc / w).astype(np.int)
    rr = (rr / w).astype(np.int)

    # The new image takes on the dimensions calculated above, and we paste in to that
    # image only in places where IMAGE2 is defined such that we don't wrap around & 
    # "repeat" IMAGE2.
    result_image = np.zeros((bottom_edge + abs(top_edge) + 1, right_edge + abs(left_edge) + 1, 3), dtype="uint8")
    overlap = np.where((cc >= 0) & (cc < image2.shape[1]) &
                   (rr >= 0) & 
                   (rr < image2.shape[0]))
    cc = cc[overlap]
    rr = rr[overlap]

    x_orig, y_orig, _ = mask
    x_orig = np.squeeze(np.asarray(x_orig))
    y_orig = np.squeeze(np.asarray(y_orig))

    x_orig = x_orig[overlap].astype(np.int)
    y_orig = y_orig[overlap].astype(np.int)

    offset_x = abs(min(left_edge, 0))
    offset_y = abs(min(top_edge, 0))

    x_orig += offset_x
    y_orig += offset_y

    image1_end_x = image.shape[1] + offset_x
    image1_end_y = image.shape[0] + offset_y

    # Paste the two images into their respective places
    result_image[y_orig, x_orig] = image2[rr, cc]
    result_image[offset_y : image1_end_y, offset_x : image1_end_x] = image

    # Blend the two images around the corners.
    alpha = np.cos(np.linspace(0, 3.141592628/2, int(image.shape[1]/2))) ** 8
    alpha = np.hstack([np.ones(int(image.shape[1]/2), dtype="float64"), alpha])
    finalAlpha = alpha
    for _ in range(image.shape[0] - 1):
        finalAlpha = np.vstack([finalAlpha, alpha])
    gray = finalAlpha.reshape((finalAlpha.shape[0], finalAlpha.shape[1], 1))
    finalAlpha = np.dstack([gray, gray, gray])

    alpha_image1 = image * finalAlpha

    result_image[offset_y : image1_end_y, offset_x : image1_end_x] = \
        alpha_image1 * finalAlpha + \
        result_image[offset_y : image1_end_y, offset_x : image1_end_x] * (1 - finalAlpha)

    return result_image

def rectify(image, H):
    """ Rectify a small square section of an image. """
    shape = image.shape
    max_x = shape[1]
    max_y = shape[0]

    bottom_left = np.matrix([[0], [max_y], [1]])
    bottom_right = np.matrix([[max_x], [max_y], [1]])
    top_left = np.matrix([[0], [0], [1]])
    top_right = np.matrix([[max_x], [0], [1]])

    corners = [bottom_left, bottom_right,
                       top_left, top_right]

    corners = [H @ point for point in corners]
    corners = [point / point[2] for point in corners]
    transformed_x_max = max(corners, key=lambda x: x[0])[0].astype(np.int)
    transformed_y_max = max(corners, key=lambda x: x[1])[1].astype(np.int)
    transformed_x_min = min(corners, key=lambda x: x[0])[0].astype(np.int)
    transformed_y_min = min(corners, key=lambda x: x[1])[1].astype(np.int)

    right_edge, bottom_edge, left_edge, top_edge = [transformed_x_max[0, 0], transformed_y_max[0, 0],
           transformed_x_min[0, 0], transformed_y_min[0, 0]]
    right_edge = max(right_edge, image.shape[1])
    bottom_edge = max(bottom_edge, image.shape[0])

    mask = draw.polygon([0, right_edge + abs(left_edge), right_edge + abs(left_edge), 0],
                        [0, 0, bottom_edge + abs(top_edge), bottom_edge + abs(top_edge)])

    mask = np.matrix(np.vstack([mask, np.ones(len(mask[0]))]))
    mask2 = np.linalg.inv(H) @ mask
    cc, rr, w = mask2

    cc = np.squeeze(np.asarray(cc))
    rr = np.squeeze(np.asarray(rr))
    w = np.squeeze(np.asarray(w))

    cc = (cc / w).astype(np.int)
    rr = (rr / w).astype(np.int)

    result_image = np.zeros((bottom_edge + abs(top_edge) + 1, right_edge + abs(left_edge) + 1, 3), dtype="uint8")
    overlap = np.where((cc >= 0) & (cc < image.shape[1]) &
                   (rr >= 0) & 
                   (rr < image.shape[0]))

    cc = cc[overlap]
    rr = rr[overlap]

    x_orig, y_orig, _ = mask
    x_orig = np.squeeze(np.asarray(x_orig))
    y_orig = np.squeeze(np.asarray(y_orig))

    x_orig = x_orig[overlap].astype(np.int)
    y_orig = y_orig[overlap].astype(np.int)

    result_image[y_orig, x_orig] = image[rr, cc]
    result_image[int(transformed_y_min):int(transformed_y_max),int(transformed_x_min):int(transformed_x_max)] = [199, 199, 199]

    return result_image


# Run a warp or rectification sequence.

def warp(image1_name, image2_name, select = False):
    left = skio.imread(jpg_name(image1_name))
    right = skio.imread(jpg_name(image2_name))

    if select:
        select_points(left, 8, image1_name)
        select_points(zion_right, 8, image2_name)

    left_points = retrieve_points(image1_name)
    right_points = retrieve_points(image2_name)

    H = computeH(left_points, right_points)
    warp = warpImage(left, right, H)
    show(warp)

def rect(image1_name, select = False):
    """ Select a square on a provided image. """
    image = skio.imread(jpg_name(image1_name))

    if select:
        select_points(image, 4, image1_name)

    image_points = retrieve_points(image1_name)

    # A square, with coordinates starting from the top left
    # and going clockwise.    
    min_dim = min(image.shape[0], image.shape[1])
    square = [[0, 0], [0, min_dim], [min_dim, min_dim], [min_dim, 0]]
    H = computeH(square, image_points)
    rectified = cv2.warpPerspective(image, H, image.shape[:2])
    #rectified = rectify(image, H)
    show(rectified)

#warp("train_left_small", "train_right_small", select = False)
#warp("martinez_left", "martinez_right", select = False)
# warp("amtrak_left", "amtrak_right", select = False)
#select_points(skio.imread(jpg_name("scenic_right")), 8, "scenic_right")
#warp("scenic_left", "scenic_right", select = False)
#display_points("train_left_small", "train_right_small")
# display_points("amtrak_left", "amtrak_right")
# display_points("martinez_left", "martinez_right")
rect("scenic_right", select = False)
#rect("train_left_small2", select = True)

#sky()