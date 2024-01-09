import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours


# Convert a mask to border image
def mask_to_border(mask):
    height, width = mask.shape
    border = np.zeros(shape=(height, width))
    # print(border)

    contours = find_contours(image=mask, level=128)
    # print(contours)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255 # give white color on the border

    return border

# Mask to bounding boxes
def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask=mask)
    lbl = label(label_image=mask)
    props = regionprops(label_image=lbl)

    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    return mask

if __name__ == '__main__':

    """ Load the dataset """
    # It will return list image paths
    images = sorted(glob(pathname=os.path.join("data", "image", "*")))
    masks = sorted(glob(pathname=os.path.join("data", "mask", "*")))

    # Create folder to save results
    if not os.path.exists(path="results"):
        os.makedirs(name="results", exist_ok=True)

    # Loop over the dataset
    for img, mask in tqdm(zip(images, masks), total=len(images)):
        # Extract the name from image
        name = img.split("\\")[-1].split(".")[0]

        # Read the images and masks
        img = cv2.imread(filename=img, flags=cv2.IMREAD_COLOR) # to read as a color image
        mask_ = cv2.imread(filename=mask, flags=cv2.IMREAD_GRAYSCALE) # to read as a gray scale image

        # border = mask_to_border(mask=mask_)

        # Detecting bounding boxes
        bboxes = mask_to_bbox(mask=mask_)

        # Making bounding box on image
        for bbox in bboxes:
            img = cv2.rectangle(img=img, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(255, 0, 0), thickness=2)

        # cv2.imshow(winname="img", mat=img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
            
        # Saving images
        cat_img = np.concatenate([img, parse_mask(mask_)], axis=1)
        cv2.imwrite(filename=f"results/b{name}.png", img=img)
        cv2.imwrite(filename=f"results/{name}.png", img=cat_img)
    
    print("Done everything")