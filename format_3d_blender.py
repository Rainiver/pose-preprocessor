# import numpy as np
# # load npz file
# npz_path = 'case_dummy/cameras.npz'
# data = np.load(npz_path)
#
# print(data)
#
# for key,value in data.items():
#     print(key)
#     print(value.shape)
#     print(value)
import os, cv2
import numpy as np
from h3ds.dataset import H3DS
from PIL import Image

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def project_scene(mesh, img, cam, color=[255, 0, 0]):

    # Expand cam attributes
    K, P = cam
    P_inv = np.linalg.inv(P)

    # Project mesh vertices into 2D
    p3d_h = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
    p2d_h = (K @ P_inv[:3, :] @ p3d_h.T).T
    p2d = p2d_h[:, :-1] / p2d_h[:, -1:]

    # Draw p2d to image
    img_proj = np.array(img)
    p2d = np.clip(p2d, 0, img.width - 1).astype(np.uint32)
    img_proj[p2d[:, 1], p2d[:, 0]] = color

    return Image.fromarray(img_proj.astype(np.uint8))


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0],
                     [0,  c, s],
                     [0, -s, c]]).astype(np.float32)


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c,  0, -s],
                     [0,  1, 0],
                     [s, 0, c]]).astype(np.float32)

def rotate_z(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c,  s, 0],
                     [-s,  c, 0],
                     [0, 0, 1]]).astype(np.float32)

def align(vertices):
    MIN_SCALE = 1.6
    SCALE = 1.8
    new_cent = (0,0,0)

    scale = max(MIN_SCALE, vertices[:, 1].max() - vertices[:, 1].min())
    vertices /= (scale / SCALE)

    cent = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices -= (cent - new_cent)

    vertices[:, 1] = -vertices[:, 1]  # flip y
    vertices[:, 2] = -vertices[:, 2]  # flip z

    # rotate x
    dy = vertices[:, 1].max() - vertices[:, 1].min()
    dz = vertices[:, 2].max() - vertices[:, 2].min()-0.2
    pts = np.matmul(rotate_x(np.arctan(dz / dy)), vertices.T).astype(np.float32)
    print('rotate x: ', np.arctan(dz / dy))
    vertices = pts.T

    # rotate y
    dz = vertices[:, 2].max() - vertices[:, 2].min()
    dx = vertices[:, 0].max() - vertices[:, 0].min()
    pts = np.matmul(rotate_y(np.arctan(dz / dx)/2), vertices.T).astype(np.float32)
    print('rotate y: ', np.arctan(dz / dx)/2)
    vertices = pts.T

    # re-scale after rotation
    scale = max(MIN_SCALE, vertices[:, 1].max() - vertices[:, 1].min())
    vertices /= (scale / SCALE)

    cent = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    new_cent1 = (0, SCALE/2, 0)
    vertices -= (cent - new_cent1)

    # rotation
    vertices = np.matmul(
        vertices,
        np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0,-1, 0]], dtype=np.float32)
    )

    return vertices, scale, cent

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--format", type=str, default=None)
    args = parser.parse_args()

    case_dir = 'nerf_body_std/humansm'
    # save_dir = 'nerf_body_1w/humansm'
    save_dir = 'tmp_project_blender1'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load mesh
    import trimesh
    mesh_path = os.path.join(case_dir, 'mesh.obj')
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    # pts = np.matmul(rotate_x(-np.pi / 2), mesh.vertices.T).astype(np.float32)
    # pts = np.matmul(rotate_z(-5*np.pi / 180), pts).astype(np.float32)
    # mesh.vertices = pts.T
    # # scale
    # mesh.vertices /= 2
    # # mesh.export('human_transformed.obj')

    # load cameras
    intrinsics_path = os.path.join(case_dir, 'intrinsics.txt')
    with open(intrinsics_path, 'r') as f:
        first_line = f.read().split('\n')[0].split(' ')
        focal = float(first_line[0])
        cx = float(first_line[1])
        cy = float(first_line[2])

        orig_img_size = 512  # cars_train has intrinsics corresponding to image size of 512 * 512

        # intrinsics = np.array(
        #     [[focal / orig_img_size, 0.00000000e+00, cx / orig_img_size],
        #      [0.00000000e+00, focal / orig_img_size, cy / orig_img_size],
        #      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        # )
        scale = 4
        h_crop = 20
        # w_crop = 190
        w_crop = 180

        # scale = 1
        # h_crop = w_crop = 0

        intrinsics = np.array(
            [[focal*scale, 0.00000000e+00, (cx - w_crop)*scale],
             [0.00000000e+00, focal*scale, (cy - h_crop)*scale],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        )

    # load images
    images = []
    cameras = []
    img_dir = os.path.join(case_dir, 'rgb')
    # img_path = os.path.join('/data/qingyao/neuralRendering/NerfRecons/data/body_sm_c1/preprocessed/mask', 'image')
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        # RGBA to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(img)
        pose_path = img_path.replace('rgb', 'pose').replace('png', 'txt')
        with open(pose_path, 'r') as f:
            pose = np.array([float(n) for n in f.read().split(' ')]).reshape(4, 4)
            # correct pose
            R = pose[:3, :3]
            t = pose[:3, 3]
            t = t * 2 # scale

            rotate_mat = rotate_x(np.pi / 2)
            R = rotate_mat @ R
            t = rotate_mat @ t
            rotate_mat = rotate_y(5*np.pi / 180)
            R = rotate_mat @ R
            t = rotate_mat @ t
            pose[:3, :3] = R
            pose[:3, 3] = t

            cameras.append((intrinsics, pose))
    n_images = len(images)
    print('n_images: ', n_images)



    cameras_computed = []
    images_computed = []
    for view_id in range(n_images):

        img = images[view_id]
        # size of img
        # img = img[h_crop:h_crop+128, w_crop:w_crop+128, :]
        img = img.crop((w_crop, h_crop, w_crop+128, h_crop+128))
        w, h = img.size
        # img = img.resize((int(w*scale), int(h*scale)), Image.ANTIALIAS)
        # resize PIL image with antialiasing
        # PIL to numpy
        img = np.array(img)
        # resize img
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        # numpy to PIL
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, '%04d_img.jpg'%view_id))

        images_computed.append(img)

        cam = cameras[view_id]
        cameras_computed.append(cam)




    # Project the mesh on each image
    os.makedirs(save_dir, exist_ok=True)
    for idx, (img, cam) in enumerate(zip(images_computed, cameras_computed)):
        img_proj = project_scene(mesh, img, cam)
        # img_proj = img_proj.resize((w, h))
        img_proj.save(os.path.join(save_dir, '%04d.jpg'%idx))
        print('save %s'%os.path.join(save_dir, '%04d.jpg'%idx))





