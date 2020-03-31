import numpy as np

def random_erasing(
        img, # Input Image
        erasing_prob=0.5, # Probability of erasing a part of the image
        erasing_area_ratio=(0.1,0.4), 
        erasing_aspect_ratio=(0,4), # The aspect ration of cut out part
        ):

    p = np.random.uniform(0,1)
    if p >= erasing_prob:
        return img
    else:
        width = img.shape[0]
        height = img.shape[1]
        area_of_image = width * height
        while True:
            se = np.random.uniform(*erasing_area_ratio)*area_of_image
            re = np.random.uniform(*erasing_aspect_ratio)

            We,He = int(np.sqrt(se*re)), int(np.sqrt(se/re))

            xe = np.random.randint(0,width)
            ye = np.random.randint(0,height)

            if xe+We <= width and ye+He <= height:
                img[xe:xe+We, ye:ye+He] = np.random.uniform(0,1,size=(We,He,img.shape[-1]))
                return img
