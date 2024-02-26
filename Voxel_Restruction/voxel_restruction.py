import pickle

import cv2
import glm
import numpy as np

from executable import main
from manually_process.utils import get_camera_extrinsic, rotate_voxels
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# def get_camera_positions():
#     # Generates dummy camera locations at the 4 corners of the room
#     cam_position = []
#     for i in range(1, 5):
#         fs = cv2.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv2.FILE_STORAGE_READ)
#         rvec, tvec = get_camera_extrinsic(i)
#         # tvec = fs.getNode('tvec').mat()
#         # rvec = fs.getNode('rvec').mat()
#         R, _ = cv2.Rodrigues(rvec)
#         R_inv = R.T
#         position = -R_inv.dot(tvec)  # get camera position
#         # get camera position in voxel space units(swap the y and z coordinates)
#         Vposition = np.array([position[0] * 3, position[2] * 3, position[1] * 3])
#         cam_position.append(Vposition)
#     color = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
#     return cam_position, color
#
#
# def get_camera_rotation_matrices():
#     cam_rotations = []
#     for i in range(4):
#         # fs = cv2.FileStorage('./data/cam{}/config.xml'.format(i+1), cv2.FILE_STORAGE_READ)
#         rvec, tvec = get_camera_extrinsic(i)
#         R, _ = cv2.Rodrigues(rvec)
#
#         R[:, 1], R[:, 2] = R[:, 2], R[:, 1].copy()  # swap y and z (exchange the second and third columns)
#         R[1, :] = -R[1, :]      # invert rotation on y (multiply second row by -1)
#         # rotation matrix: rotation 90 degree about the y
#         rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
#         R = np.matmul(R, rot)
#
#         # convert to mat4x4 format
#         RM = np.eye(4)
#         RM [:3, :3] = R
#         RM  = glm.mat4(*RM .flatten())
#         cam_rotations.append(RM)
#     print(cam_rotations)
#     return cam_rotations
#
#
# def get_data_and_colors(width, height, depth, bg, press_num):
#     # Initializes the voxel positions
#     prevForeground = [None for _ in range(4)]
#     lookup = None
#     width = 25
#     height = 25
#     depth = 25
#
#     voxel_size = 0.4
#     data0 = []
#     colors = []
#     for x in np.arange(-3, width - 3, voxel_size):
#         for y in np.arange(-2, height - 2, voxel_size):
#             for z in np.arange(-3, depth - 3, voxel_size):
#                 data0.append([x, y, z])
#
#     # flags to save data and 2D coordinates
#     flags = [[[[], []] for _ in range(len(data0))] for _ in range(4)]
#     data0 = np.float32(data0)
#     # first frame, compute lookup table
#     if press_num <= 2726:
#         for i in range(4):
#             # foreground = bg[i]
#             foreground = cv2.imread(bg[i])
#             # fs = cv2.FileStorage('./data/cam{}/config.xml'.format(i + 1), cv2.FILE_STORAGE_READ)
#             newpath = bg[i].replace('video/Take30', 'extrinsics/Take25')
#             newpath = newpath[:38] + 'config.xml'
#             fs = cv2.FileStorage(newpath, cv2.FILE_STORAGE_READ)
#             mtx = fs.getNode('mtx').mat()
#             dist = fs.getNode('dist').mat()
#             rvec = fs.getNode('rvec').mat()
#             tvec = fs.getNode('tvec').mat()
#
#             pts, jac = cv2.projectPoints(data0, rvec, tvec, mtx, dist)
#             pts = np.int32(pts)
#
#             for j in range(len(data0)):
#                 # cv2.circle(foreground, tuple([pts[j][0][0], pts[j][0][1]]), 1, (0, 0, 255), -1)
#                 try:
#                     # print(foreground[pts[j][0][1]][pts[j][0][0]].sum())
#                     if foreground[pts[j][0][1]][pts[j][0][0]].sum() == 0:   # if point falls into the background
#                         flags[i][j] = [0, [pts[j][0][1], pts[j][0][0]]]
#                     else:
#                         flags[i][j] = [1, [pts[j][0][1], pts[j][0][0]]]
#                 except:
#                     # print("Out of range!")
#                     flags[i][j] = [0, [pts[j][0][1], pts[j][0][0]]]
#                     continue
#             prevForeground[i] = foreground
#             # cv2.imshow('foreground', foreground)
#             # cv2.waitKey(0)
#         lookup = flags
#     cv2.destroyAllWindows()
#     # print(lookup)
#     data = []
#     columnSum = np.zeros(len(data0))
#     colorpath = 'data/cam2/segment/video.jpg'
#     # clip = cv2.imread('./data/cam{}/video.jpg'.format(2))
#     clip = cv2.imread(colorpath)
#     for i in range(len(data0)):
#         for j in range(len(lookup)):
#             columnSum[i] += lookup[j][i][0]
#
#     # if voxels in all views are visible, show it on the screen
#     for i in range(len(data0)):
#         if columnSum[i] == 4:
#             data.append(data0[i])
#             # color.append(colors[i])
#             colors.append(clip[lookup[1][i][1][0]][lookup[1][i][1][1]] / 256)
#
#     #saveCoord(data, bg)
#     # rotate array -90 degree along the x-axis.
#     Rx = np.array([[1, 0, 0],
#                    [0, 0, 1],
#                    [0, -1, 0]])
#     dataR = [Rx.dot(p) for p in data]
#     dataR = [np.multiply(DR, 5) for DR in dataR]
#     return dataR, colors


def draw_mesh(positions):
    voxel = np.int32(positions)
    width = np.max(voxel[:, 0]) - np.min(voxel[:, 0])
    depth = np.max(voxel[:, 1]) - np.min(voxel[:, 1])
    height = np.max(voxel[:, 2]) - np.min(voxel[:, 2])

    grid = np.zeros((width + 1, height + 1, depth + 1), dtype=bool)

    print(grid.shape)

    for i in range(len(voxel)):
        grid[voxel[i][0] - np.min(voxel[:, 0])][1 - (voxel[i][2] - np.min(voxel[:, 2]))][
            voxel[i][1] - np.min(voxel[:, 1])] = True

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(grid, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, int(1.5 * width))
    ax.set_ylim(0, int(1.5 * height))
    ax.set_zlim(0, int(1.5 * depth))

    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def generate_voxel_map(config):
    # main()
    # read data
    with open('data_300.pkl', 'rb') as f:
        data = pickle.load(f)
    data = rotate_voxels(data, np.radians(90), 'x')

    draw_mesh(np.array(data))

