import numpy as np
# import cv2
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
# import requests

# from io import BytesIO
# import base64

# def visualize_attention(img_url, attention_mask, title, cmap="jet"):
def visualize_attention(img_path, attention_mask, title, output_path, mask_size, desired_size=512, cmap="jet"):
	"""
	img_path: read the path of image
	attention_mask: 2-D numpy array
	cmap: style of attention map
	"""
	# print("load image from:", img_path)
	# # load the image
	# img = Image.open(img_path, mode="r")

	# img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	# img = Image.open(requests.get(img_url, stream=True).raw) # 640x480
	img = Image.open(img_path, mode="r")
	# img = Image.open(BytesIO(base64.urlsafe_b64decode(img_path))).convert('RGB')
	img_w, img_h = img.size[0], img.size[1]
	plt.subplots(nrows=1, ncols=1, figsize=(0.02*img_h, 0.02*img_w))

	# # desired image size
	# desired_size = 512

	# resize image if shorter edge is smaller than the desired image size
	if img_w > img_h and img_h < desired_size:
		ratio = desired_size/float(img_h)
		img_h = desired_size
		img_w = int(math.ceil(img_w*ratio))
		img = img.resize((img_w, img_h), Image.BICUBIC)
	elif img_w < img_h and img_w < desired_size:
		ratio = desired_size/float(img_w)
		img_w = desired_size
		img_h = int(math.ceil(img_h*ratio))
		img = img.resize((img_w, img_h), Image.BICUBIC)

	# scale image
	if img_w > img_h:
		crop_size = img_h
	else:
		crop_size = img_w

	# crop image to short edge
	left = int((img_w-crop_size)/2)
	right = int((img_w-crop_size)/2+crop_size)
	upper = int((img_h-crop_size)/2)
	bottom = int((img_h-crop_size)/2+crop_size)
	cropped_img = img.crop((left, upper, right, bottom))

	# resize image to 256x256
	cropped_img = cropped_img.resize((desired_size, desired_size), Image.BICUBIC)

	# plot
	plt.imshow(cropped_img, alpha=1)
	plt.axis("off")

	# print(attention_mask.shape[0])
	# assert False

	# normalize the attention mask
	# mask = cv2.resize(attention_mask, (size, size), interpolation=cv2.INTER_NEAREST)
	# mask = np.repeat(attention_mask, 256/attention_mask.shape[0], axis=0)
	# mask = np.repeat(mask, 256/attention_mask.shape[1], axis=1)
	mask = np.repeat(attention_mask, desired_size/attention_mask.shape[0], axis=0)
	mask = np.repeat(mask, desired_size/attention_mask.shape[1], axis=1)
	# np.savetxt("mask_new.txt", mask)
	# assert False

	normed_mask = mask/mask.max()
	normed_mask[normed_mask<0] = 0
	# print(normed_mask.min())
	# print(mask[mask>1])
	# normed_mask = mask
	normed_mask = (normed_mask*255).astype("uint8")
	# normed_mask = (normed_mask*127.5+127.5).astype("uint8")
	plt.imshow(normed_mask, alpha=0.5, interpolation="nearest", cmap=cmap)

	plt.title(title)

	# plt.savefig("./output256/{}.png".format(title))
	plt.savefig(os.path.join(output_path, "{}.png".format(title)))
	# plt.show()


if __name__ == '__main__':
	img_path = "./coco_karpathy_split/validation_set_part/image/000000001448.jpg"

	# attn_path = ""
	
	# attention_mask = np.load(attn_path)
	# visualize_attention(img_path, attention_mask, cmap="jet")

