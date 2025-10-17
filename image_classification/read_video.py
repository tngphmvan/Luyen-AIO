import os
import cv2
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def _doc_video_image_folder(input_folder: str, output_folder: str):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in tqdm(os.walk(input_folder), desc="Processing folders"):
        for file in tqdm(files, desc="Processing files", leave=False):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)

                if image is None:
                    continue

                if os.path.abspath(root).lower().count("live") != 0:
                    img_class = "live"
                elif os.path.abspath(root).lower().count("cutout") != 0:
                    img_class = "cutout"
                elif os.path.abspath(root).lower().count("mask") != 0:
                    img_class = "mask"
                elif os.path.abspath(root).lower().count("mask3d") != 0:
                    img_class = "mask3d"
                elif os.path.abspath(root).lower().count("monitor") != 0:
                    img_class = "monitor"
                elif os.path.abspath(root).lower().count("outline") != 0:
                    img_class = "outline"
                elif os.path.abspath(root).lower().count("outline3d") != 0:
                    img_class = "outline3d"
                elif os.path.abspath(root).lower().count("PC_Replay") != 0:
                    img_class = "PC_Replay"
                elif os.path.abspath(root).lower().count("Print_Attacks_Samples") != 0:
                    img_class = "Print_Attacks_Samples"
                elif os.path.abspath(root).lower().count("Smartphone_Replay") != 0:
                    img_class = "Smartphone_Replay"
                os.makedirs(output_subfolder, exist_ok=True)

                output_path = f"{os.path.join(output_subfolder, file)}.jpg"
                cv2.imwrite(output_path, image)
            if file.lower().endswith(('.mov', '.mp4', '.avi', '.mkv')):
                cap = cv2.VideoCapture(os.path.join(root, file))
                if not cap.isOpened():
                    continue
                total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frame//2)
                flag, frame = cap.read()
                if flag:
                    if os.path.abspath(root).lower().count("live") != 0:
                        img_class = "live"
                    elif os.path.abspath(root).lower().count("cutout") != 0:
                        img_class = "cutout"
                    elif os.path.abspath(root).lower().count("mask") != 0:
                        img_class = "mask"
                    elif os.path.abspath(root).lower().count("mask3d") != 0:
                        img_class = "mask3d"
                    elif os.path.abspath(root).lower().count("monitor") != 0:
                        img_class = "monitor"
                    elif os.path.abspath(root).lower().count("outline") != 0:
                        img_class = "outline"
                    elif os.path.abspath(root).lower().count("outline3d") != 0:
                        img_class = "outline3d"
                    elif os.path.abspath(root).lower().count("PC_Replay") != 0:
                        img_class = "PC_Replay"
                    elif os.path.abspath(root).lower().count("Print_Attacks_Samples") != 0:
                        img_class = "Print_Attacks_Samples"
                    elif os.path.abspath(root).lower().count("Smartphone_Replay") != 0:
                        img_class = "Smartphone_Replay"
                    output_subfolder = os.path.join(output_folder, img_class)
                    os.makedirs(output_subfolder, exist_ok=True)

                    output_path = f"{os.path.join(output_subfolder, file)}.jpg"
                    cv2.imwrite(output_path, frame)
                cap.release()


_doc_video_image_folder(
    r"D:\publics_data_train\publics_data_train", "data")

# dataset = ImageFolder("data")
# dataloader = DataLoader(dataset=dataset, batch_size=64, num_workers=4)
# print(dataloader)
