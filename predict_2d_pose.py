"""
Script for predicting 2D human pose using OpenPose library wrapper.

Author: Ilya Petrov
"""
import sys
sys.path.append('/usr/local/python')
import glob
import json
import argparse
import cv2
import numpy as np
import pyopenpose as op
import os


def preset_params(args):
    params = dict()

    params["body"] = 1 if 'b' in args.mode else 0

    if 'h' in args.mode:
        params["hand"] = True
        # params["hand_net_resolution"] = "1312x736"
        params["hand_scale_number"] = 6
        params["hand_scale_range"] = 0.4
        params["hand_render_threshold"] = 0.01
    else:
        params["hand"] = False

    if 'f' in args.mode:
        params["face"] = True
        params["face_net_resolution"] = "480x480"
        params["face_render_threshold"] = 0.01
    else:
        params["face"] = False

    return params


def filter_background_detections(detections):
    if detections is not None and detections.ndim == 3:
        mean_confidence = np.mean(detections[:, :, 2], axis=1)
        index = np.argmax(mean_confidence)

        return detections[index].tolist()
    else:
        return []


def step2(args):
    input_folder = args.render_dir
    # print(args.input_folder)
    # input_folder = Path(args.input_folder) / '/'
    # print(input_folder)

    results_folder = os.path.dirname(input_folder)
    # custom params for the model (refer to include/openpose/flags.hpp for more parameters)
    op_params = dict()
    ### PATH
    op_params["model_folder"] = "/data/eas/3rd/openpose/models/"
    op_params["net_resolution"] = "-1x176"
    # op_params["net_resolution"] = "720x480"

    # op_params["scale_number"] = 3
    # op_params["scale_gap"] = 0.25
    op_params.update(preset_params(args))

    # start OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(op_params)
    op_wrapper.start()

    # print(input_folder)
    # print(list(input_folder.glob(f"*.jpg")))
    # # list input images
    # input_image_paths = []
    # for ext in ["jpg", "jpeg", "png"]:
    #     input_image_paths.extend(list(input_folder.glob(f"*.{ext}")))
    # input_image_paths = sorted(input_image_paths)
    input_image_paths = sorted(glob.glob(input_folder+'/*.jpg'))
    print('found %d images'%len(input_image_paths))

    # create input Datums and get predictions
    op_vector_datum = []
    i = 0
    for input_image_path in input_image_paths:
        # if i>0:
        #     continue
        op_datum = op.Datum()
        image = cv2.imread(str(input_image_path))
        op_datum.cvInputData = image

        op_wrapper.emplaceAndPop(op.VectorDatum([op_datum]))
        op_vector_datum.append(op_datum)
        i+=1

    # convert OP datum to internal results format
    # results structure: {<filename>: [<list with OP predictions>]}
    results = {}
    for input_image_path, datum in zip(input_image_paths, op_vector_datum):
        input_name = os.path.basename(input_image_path)[:-4]
        results[input_name] = {
            "pose_keypoints_2d": filter_background_detections(datum.getPoseKeypoints()),
            "face_keypoints_2d": filter_background_detections(datum.getFaceKeypoints()),
            "hand_left_keypoints_2d": filter_background_detections(datum.getHandKeypointsL()),
            "hand_right_keypoints_2d": filter_background_detections(datum.getHandKeypointsR())
        }

        # optionally save visualisations
        outdir = os.path.join(results_folder, "2D_pose_vis")
        os.makedirs(outdir, exist_ok=True)
        if args.visualize:
            vis_result_path = os.path.join(outdir, input_name+'.jpg')
            cv2.imwrite(str(vis_result_path), datum.cvOutputData)

    # save results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    save_path = os.path.join(results_folder, "2D_pose.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)
    print('save kp to %s'%save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenPose predictor.")
    parser.add_argument("--render_dir", type=str,
                        help="Path to folder with data")
    parser.add_argument("--mode", "-m", type=str, choices=['b', 'h', 'f'], nargs='+', default=['b'],
                        help="Switching between detecting body, hand, and face joints, "
                             "modes can be combined (default: b h)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Save visualizations (default: False)")
    args = parser.parse_args()



    step2(args)
