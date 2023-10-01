try:
    import traceback
    import os
    import numpy as np
    from pathlib import Path
    
    import cv2
    import anodet
    import torch
    from torch.utils.data import DataLoader    

except:
    traceback.print_exc()
    
if 'anodet' in locals():
    print("Hello from anodet")

def init(s=None):
    global start
    if s is not None:
        start = s 
    load()

def load():
    global start
    
    global mean, cov_inv, padim    
    global MODEL_DATA_PATH, MODEL_NAME, DATASET_PATH
    
    base_model_path = os.path.join("Profile",start.Config.profile)
    
    if str(start.Config.model_path) != ".":
        base_model_path = start.Config.model_path
    
    MODEL_DATA_PATH = Path(base_model_path,"ANO_MODEL")
    MODEL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    MODEL_NAME = "ano"
    
    DATASET_PATH = Path(base_model_path,"ano_images")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    mean = None
    cov_inv = None
    padim = None
    
    try:
        if Path(MODEL_DATA_PATH, MODEL_NAME+"_mean.pt").is_file():
            mean = torch.load(Path(MODEL_DATA_PATH, MODEL_NAME+"_mean.pt"))
            cov_inv = torch.load(Path(MODEL_DATA_PATH, MODEL_NAME+"_cov_inv.pt")) 
            padim = anodet.Padim(backbone='resnet18', mean=mean, cov_inv=cov_inv, device=torch.device('cpu'))
    except:
        traceback.print_exc()
        
def train():
    global mean, cov_inv, padim    
    global MODEL_DATA_PATH, MODEL_NAME, DATASET_PATH
    
    dataset = anodet.AnodetDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=1)
    print("Number of images in dataset:", len(dataloader.dataset))
    
    padim = anodet.Padim(backbone='resnet18')#wide_resnet50)
    
    padim.fit(dataloader)
    
    mean = padim.mean
    cov_inv = padim.cov_inv
    
    torch.save(padim.mean, Path(MODEL_DATA_PATH, MODEL_NAME+"_mean.pt"))
    torch.save(padim.cov_inv, os.path.join(MODEL_DATA_PATH, MODEL_NAME+"_cov_inv.pt"))
    
def inference(images, thresh):
    global mean, cov_inv, padim

    batch = anodet.to_batch(images, anodet.standard_image_transform, torch.device('cpu'))
    
    if mean is None:    
        mean = torch.load(os.path.join(MODEL_DATA_PATH,  MODEL_NAME+"_mean.pt"))
        cov_inv = torch.load(os.path.join(MODEL_DATA_PATH,  MODEL_NAME+"_cov_inv.pt")) 
        padim = anodet.Padim(backbone='resnet18', mean=mean, cov_inv=cov_inv, device=torch.device('cpu'))
        
    image_scores, score_maps = padim.predict(batch)    
    
    score_map_classifications = anodet.classification(score_maps, thresh)
    image_classifications = anodet.classification(image_scores, thresh)
    print("Image scores:", image_scores)
    #print("Image classifications:", image_classifications)
    
    test_images = np.array(images).copy()
    
    boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=5)
    heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = anodet.visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))
    
    boundary_image = cv2.cvtColor(boundary_images[0], cv2.COLOR_RGB2BGR)
    heatmap_image = cv2.cvtColor(heatmap_images[0], cv2.COLOR_RGB2BGR)
    
    output_image = cv2.hconcat([boundary_image, heatmap_image])
    
    # for idx in range(len(images)):
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("status", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("area", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("highlight", cv2.WINDOW_NORMAL)
        # cv2.imshow("image" , cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR))
        # cv2.imshow("status" , cv2.cvtColor(boundary_images[idx], cv2.COLOR_RGB2BGR))
        # cv2.imshow("area" , cv2.cvtColor(heatmap_images[idx], cv2.COLOR_RGB2BGR))
        # cv2.imshow("highlight" , cv2.cvtColor(highlighted_images[idx], cv2.COLOR_RGB2BGR))

    return image_scores, output_image