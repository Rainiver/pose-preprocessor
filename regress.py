import torch
import cv2
from PIL import Image
import numpy as np
import imageio
import trimesh
import json
import os
import math

# from lib.smpl.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplx import SMPL
from body_objectives import torch_pose_obj_data
from torch_functions import batch_sparse_dense_matmul
from scipy.spatial.transform import Rotation


def project_scene(mesh, intrinsics, c2w, color=[255, 0, 0]):
    # Expand cam attributes
    K = intrinsics  # (3,3)
    P = c2w  # (4,4)
    # K, P = cam
    # cam = (intrinsics, cam2world_matrix)

    P_inv = np.linalg.inv(P)

    # Project mesh vertices into 2D
    # print('mesh.vertices.shape', mesh.vertices.shape)   # (6890, 3)
    p3d_h = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
    # print('p3d_h.shape', p3d_h.shape) (6890,4)
    p2d_h = (K @ P_inv[:3, :] @ p3d_h.T).T
    p2d = p2d_h[:, :-1] / p2d_h[:, -1:]

    # Draw p2d to image
    # img_proj = np.array(img)
    width = 512
    img_proj = np.zeros((512, 512, 3), dtype=np.uint8)  # 纯黑图像
    p2d = np.clip(p2d, 0, width - 1).astype(np.uint32)
    img_proj[p2d[:, 1], p2d[:, 0]] = color
    # img_proj = cv2.undistort(img_proj, K, D)

    return Image.fromarray(img_proj.astype(np.uint8))


def project_points(p3d, intrinsics, c2w, color=[255, 0, 0]):
    # Expand cam attributes
    K = intrinsics  # (3,3)
    P = c2w  # (4,4)
    # K, P = cam
    # cam = (intrinsics, cam2world_matrix)

    P_inv = np.linalg.inv(P)

    # Project 3D keypoints into 2D
    # p3d_h = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
    # print('p3d.shape', p3d.shape)
    p3d_h = np.hstack((p3d, np.ones((p3d.shape[0], 1))))
    p2d_h = (K @ P_inv[:3, :] @ p3d_h.T).T
    p2d = p2d_h[:, :-1] / p2d_h[:, -1:]

    # Draw p2d to image
    # img_proj = np.array(img)
    img = np.zeros((512, 512, 3), dtype=np.uint8)  # 纯黑图像
    width = 512
    p2d = np.clip(p2d, 0, width - 1).astype(np.uint32)
    # img_proj[p2d[:, 1], p2d[:, 0]] = color
    # img_proj = cv2.undistort(img_proj, K, D)

    return p2d
    # return Image.fromarray(img_proj.astype(np.uint8))


body_model = SMPL('/data/vdd/zhongyuhe/workshop/AG3D/training/deformers/smplx/SMPLX', gender='neutral')

# print('J', J)
# print('face', face)
# print('hands', hands)
path = '/data/vdd/zhongyuhe/workshop/GauHuman/data/zju_mocap_refine/my_392/'

ann_file = os.path.join(path, 'annots.npy')
annots = np.load(ann_file, allow_pickle=True).item()
cams = annots['cams']
# print('cams[K].shape', len(cams['K']))  # 23个相机

# K = np.array(cams['K'][0])
# print('K', K)
#
# D = np.array(cams['D'][0])  # distortion
# print('D', D)
#
# R = np.array(cams['R'][0])
# print('R', R)
# T = np.array(cams['T'][0]) / 1000.
# print('T', T)
#
# # porject 3d keypoints to 2d keypoints
# intrinsics = np.array(K, dtype=np.float32).reshape(3, 3)


# c2w = np.eye(4)
# c2w[:3, :3] = R
# c2w[:3, 3:4] = T
#
# c2w = np.array(
#             [[-1., 0., 0., 0.5],
#              [0., 1., 0., 1.],
#              [0., 0., -1., -100.],
#               [0., 0., 0., 1.]], dtype=np.float32
#         )
c2w = np.array(
    [[-1., 0., 0., 0.],
     [0., 1., 0., 0.],
     [0., 0., 1., -100.],
     [0., 0., 0., 1.]], dtype=np.float32
)
#
# c2w = np.array(
#             [[1., 0., 0., 0.],
#              [0., 0., 1., 0.],
#              [0., -1., 0., -100.],
#               [0., 0., 0., 1.]], dtype=np.float32
#         )
# focal_length = 2500
# print('fx, fy, cx, cy: ', fx, fy, cx, cy)


orig_img_size = 512
# fx = 40
# fy = 40
fx = 50
fy = 50
cx = 0.5
cy = 0.5

intrinsics = np.array(
    [[fx * orig_img_size, 0.00000000e+00, cx * orig_img_size],
     [0.00000000e+00, fy * orig_img_size, cy * orig_img_size],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)
# intrinsics = np.array([[550, 0, 256], [0, 550, 256], [0, 0, 1]], np.float32)

# c2w = np.array([-0.994, 0.094, -0.062,
#                 -0.086, -0.280, 0.956,
#                 0.072, 0.955, 0.287, ], dtype=np.float32)


dist_coeffs = np.zeros((5, 1), np.float32)

root_path = '/data/vdd/zhongyuhe/workshop/GauHuman/data/zju_mocap_refine/my_392/easymocap/output-smpl-3d/smplfull/'
# root_path = 'smplfull/'
smpl_path = root_path + '000000.json'

# root_path = '/data/vdd/zhongyuhe/workshop/dataset/motion/demo_THuman/'
# npz_path = root_path + 'pose_00.npz'
# input_data = np.load("ubc_pose_dist.npy")
# input_data = np.load(npz_path, allow_pickle=True)
# num = 2500
# save_dir = 'visualization_thuman'
save_dir = 'visualization_op2'

global_orient = np.array([[3.0422, -0.0334, 0.0688],
                         [3.0940, -0.0442, 0.9080],
                         [2.7520, -0.1350, 1.6863],
                         [2.2372, -0.1320, 2.3201],
                         [1.7902, -0.0621, 2.6401],
                         [0.7323, -0.1798, 3.0502],
                         [-2.7939e-03, -3.4085e-01, 3.1297e+00]],
                         dtype=np.float32)
transl = np.array([-5.2587111e-03, 9.5774055e-02, 2.0827456e+01], dtype=np.float32).reshape(1, 3)
for file in sorted(os.listdir(root_path)):
    smpl_path = root_path + file
    with open(smpl_path) as f:
        smpl_data = json.load(f)
    for i in range(7):
        # root_orient = Rotation.from_rotvec(np.array(smpl_data["annots"][0]['Rh']).reshape([-1])).as_matrix()
        # new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
        # print('new_root_orient.shape', new_root_orient.shape)
        # global_orient = np.array(smpl_data["annots"][0]["poses"][:3])
        # global_orient = np.array([3.0422, -0.0334,  0.0688], dtype=np.float32).reshape(1, 3)
        # print('global_orient.shape', global_orient.shape)
        # print('pose.shape', np.array(smpl_data["annots"][0]["poses"]).shape)

        smpl_outputs = body_model(
            betas=torch.from_numpy(np.array(smpl_data["annots"][0]["shapes"][0], dtype=np.float32)).reshape(-1, 10),
            body_pose=torch.from_numpy(np.array(smpl_data["annots"][0]["poses"][0][3:], dtype=np.float32)).reshape(-1,
                                                                                                                   69),
            # global_orient=torch.from_numpy(np.array(smpl_data["annots"][0]["poses"][0][:3], dtype=np.float32)).reshape(-1, 3))
            # global_orient=torch.from_numpy(new_root_orient),
            global_orient=torch.from_numpy(global_orient[i]).reshape(1, 3),
            # transl=torch.from_numpy(np.array(smpl_data["annots"][0]["Th"][0], dtype=np.float32)).reshape(-1, 3) )
            transl=torch.from_numpy(transl))

        smpl_v = smpl_outputs['vertices'].clone().reshape(-1, 3)
        # smpl_v[:, [1, 2]] = smpl_v[:, [2, 1]]*(-1)  # 坐标系转换
        mesh = trimesh.Trimesh(smpl_v, body_model.faces)
        mesh.export(f"zju_mesh_origin/human_{file}.obj")

        # print('verts.shape', verts.shape)  # (1, 6890, 3)
        # smpl_outputs['vertices'][:, :, [1, 2]] = smpl_outputs['vertices'][:, :, [2, 1]]*(-1)  # 坐标系转换
        verts = smpl_outputs['vertices']
        # verts = smpl_v

        body25_reg_torch, face_reg_torch, hand_reg_torch = \
            torch_pose_obj_data(batch_size=1)

        J = batch_sparse_dense_matmul(body25_reg_torch, verts)  # 1, 25, 3
        face = batch_sparse_dense_matmul(face_reg_torch, verts)  # 1,70,3
        hands = batch_sparse_dense_matmul(hand_reg_torch, verts)  # 1,42,3

        # root = '/data/vdd/zhongyuhe/workshop/GauHuman/data/zju_mocap_refine/my_392/images/00'
        # root = 'openpose_front'
        root = 'op_7'
        file_name = '000000.jpg'
        # json_name = file.split('.')[-2] + '_keypoints.json'
        json_name = file.split('.')[-2] + f'_{i}.json'
        json_data = {"version": 1.3,
                     "people": [
                         {"person_id": [-1],
                          "pose_keypoints_2d": [],
                          "face_keypoints_2d": [],
                          "hand_left_keypoints_2d": [],
                          "hand_right_keypoints_2d": [],
                          "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [],
                          "hand_right_keypoints_3d": []}]}

        # img_path = os.path.join(root, file_name)
        # img = Image.open(img_path)
        # img_proj = project_scene(mesh, intrinsics, c2w)

        # p2d = project_points(p3d, img, intrinsics, c2w)
        J_2d = project_points(np.array(J).reshape(25, 3), intrinsics, c2w)
        # J_2d.save(os.path.join(save_dir, file.split('.')[-2] + '_J.png'))

        face_2d = project_points(np.array(face).reshape(70, 3), intrinsics, c2w)
        # face_2d.save(os.path.join(save_dir, file.split('.')[-2] + '_face.png'))
        hands_2d = project_points(np.array(hands).reshape(42, 3), intrinsics, c2w)
        # hands_2d.save(os.path.join(save_dir, file.split('.')[-2] + '_hands.png'))

        json_data["people"][0]["pose_keypoints_2d"] = J_2d.reshape(1, 50).tolist()  # (25, 2)
        json_data["people"][0]["face_keypoints_2d"] = face_2d.reshape(1, 140).tolist()  # (70, 2)
        json_data["people"][0]["hand_left_keypoints_2d"] = hands_2d[:21, :].reshape(1, 42).tolist()  # (21, 2)
        json_data["people"][0]["hand_right_keypoints_2d"] = hands_2d[21:, :].reshape(1, 42).tolist()  # (21, 2)

        with open(os.path.join(root, json_name), 'w') as write_f:
            json.dump(json_data, write_f, indent=4, ensure_ascii=False)

        # img_proj = cv2.undistort(img_proj, K, D)  # 最后一步矫正畸变
        # img_proj.save(os.path.join(save_dir, file.split('.')[-2] + f'_{i}.png'))
        print('save %s' % os.path.join(save_dir, file.split('.')[-2] + f'_{i}.png'))
