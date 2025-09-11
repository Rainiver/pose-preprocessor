import argparse
import cv2
import os
from utils.pose_utils import draw_openpose_skeleton, load_openpose_keypoints
from utils.image_utils import concat_images


def main():
    parser = argparse.ArgumentParser(description="Draw OpenPose skeletons on images.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or folder.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output image.")
    parser.add_argument("--concat", action="store_true",
                        help="If set, concatenate multiple input images horizontally.")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        images, keypoints = [], []
        for file in sorted(os.listdir(args.input)):
            if file.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(args.input, file)
                kp_path = img_path.replace(".jpg", "_keypoints.json").replace(".png", "_keypoints.json")
                image = cv2.imread(img_path)
                images.append(image)
                keypoints.append(load_openpose_keypoints(kp_path))

        drawn_images = [draw_openpose_skeleton(img, kp) for img, kp in zip(images, keypoints)]
        if args.concat:
            output_img = concat_images(drawn_images)
        else:
            output_img = drawn_images[0]

    else:  # single image
        image = cv2.imread(args.input)
        kp_path = args.input.replace(".jpg", "_keypoints.json").replace(".png", "_keypoints.json")
        keypoints = load_openpose_keypoints(kp_path)
        output_img = draw_openpose_skeleton(image, keypoints)

    cv2.imwrite(args.output, output_img)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
