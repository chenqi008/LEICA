from models.min_dalle import MinDalle
import torch
from PIL import Image
import numpy as np
import  cv2
from dalle_pytorch import VQGanVAE

vae = VQGanVAE(vqgan_config_path='checkpoints/min_dalle/model.yaml', vqgan_model_path='checkpoints/min_dalle/last.ckpt').cuda()


mindalle = MinDalle(
    models_root='./pretrained',
    dtype=torch.float16,
    device='cuda',
    is_mega=True, 
    is_reusable=True
)

  
mindalle.detokenizer.decoder = vae.model.decoder

images = mindalle.generate_images(
    text='An armchair in the shape of an avocado. An armchair imitating an avocado',
    seed=-1,
    grid_size=3,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=16,
    is_verbose=False
)
images = images.to('cpu').numpy()
for i in range(len(images)):
    image = Image.fromarray((images[i]).astype(np.uint8))
    image.save('image_{}.png'.format(i))