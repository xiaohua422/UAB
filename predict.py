import time
import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    # Modify self.colors in __init__ to change segmentation colors
    # -------------------------------------------------------------------------#
    deeplab = DeeplabV3()

    # -------------------------------------------------------------------------#
    # mode:
    # 'predict'       single image prediction
    # 'video'         video or camera detection
    # 'fps'           test model FPS
    # 'dir_predict'   batch predict images in a folder
    # 'export_onnx'   export model to ONNX format
    # -------------------------------------------------------------------------#
    mode = "dir_predict"

    # -------------------------------------------------------------------------#
    # count: enable pixel counting (area ratio calculation)
    # name_classes: class names for segmentation
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["_background_", "L1", "L2", "L3", "L4", "L5",
                    "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1", "CSF"]

    # -------------------------------------------------------------------------#
    # video settings
    # -------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    # -------------------------------------------------------------------------#
    # FPS test settings
    # -------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"

    # -------------------------------------------------------------------------#
    # batch prediction folder paths
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "phase_img_out/"

    # -------------------------------------------------------------------------#
    # ONNX export settings
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input('Input image filename: ')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read camera or video.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(deeplab.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % fps)
            frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to: " + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(f'{tact_time} seconds, {1 / tact_time} FPS @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        mask_save_path = os.path.join('phase_mask')

        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)

                # Save segmentation mask
                mask = deeplab.get_miou_png(image)
                mask_name = os.path.splitext(img_name)[0] + ".png"
                mask.save(os.path.join(mask_save_path, mask_name))

                # Save visualization result
                r_image = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify correct mode: 'predict', 'video', 'fps', 'dir_predict'.")
