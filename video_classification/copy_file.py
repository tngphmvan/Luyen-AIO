import shutil
import os

src_folder = "data/train"
dst_folder = "data/backup"

for filename in os.listdir(src_folder):
    src = os.path.join(src_folder, filename)
    dst = os.path.join(dst_folder, filename)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
