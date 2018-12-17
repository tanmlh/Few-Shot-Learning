import os
import cv2
import Function

class VideoDataset:
    """
    A dataset which contains the images of several video dir
    vid_dir: the paths of dirs containing sets of images
    class_id: the class id of these images
    """
    def __init__(self, vid_dirs, class_id, root_dir='../../dataset/COX face dataset/data/cropped_video',
                 img_size=(112, 112)):
        dirs = []
        for vid_dir in vid_dirs:
            cur_dir = os.path.join(root_dir, vid_dir)
            dirs += [os.path.join(cur_dir, f) for f in os.listdir(os.path.join(root_dir, vid_dir))]

        self.dirs = dirs
        self.class_id = class_id
        self.img_size = img_size
        self.root_dir = root_dir

    def __getitem__(self, idx):
        img_np = cv2.imread(self.dirs[idx])
        img_np = Function.resize_pad(img_np, self.img_size)
        return Function.np2nd(img_np)[0], self.class_id

    def __len__(self):
        return sum([len(dir) for dir in self.dirs])
