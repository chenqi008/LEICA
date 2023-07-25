import os
import base64
from io import BytesIO
from PIL import Image
import argparse
from tqdm import tqdm

def data_process(data_path, custom_data_filename):
    textfile = os.path.join(data_path, "text", "caption.txt")

    # handle caption
    with open(textfile, "r") as f:
        captions = f.readlines()

    item_list = [] # each item: uniq-id(for text), image-id, text, and image base64 string
    for cap_id in range(len(captions)):
        item_list.append([cap_id+1])
        item_list[cap_id].extend(captions[cap_id].replace("\n", "").split("\t"))

    # handle image
    # for item in tqdm(item_list[:20]):
    for item in tqdm(item_list):
        # imagefile = os.path.join(data_path, "image", "{}.jpg".format(item[1].zfill(12)))
        imagefile = os.path.join(data_path, "image", "{}.jpg".format(item[1].zfill(12)))
        
        img = Image.open(imagefile)

        w, h = img.size
        if w > h:
            crop_size = h
        else:
            crop_size = w

        # crop_size = 256

        # crop image to short edge
        left = int((w-crop_size)/2)
        right = int((w-crop_size)/2+crop_size)
        upper = int((h-crop_size)/2)
        bottom = int((h-crop_size)/2+crop_size)
        cropped_img = img.crop((left, upper, right, bottom))

        # print((left, upper, right, bottom))
        # print(cropped_img.size)
        # assert False

        # resize image to 256x256
        cropped_img = cropped_img.resize((256, 256), Image.LANCZOS)

        # # save the resized image to check
        # cropped_img.save(os.path.join(data_path, "image", "{}-resize-BICUBIC.jpg".format(item[1].zfill(12))))

        # convert image pattern to base64
        output_buffer = BytesIO()
        cropped_img.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        base64data = base64.urlsafe_b64encode(byte_data)
        # base64data = base64.b64encode(byte_data)
        base64data = str(base64data, 'utf-8')
        item.append(base64data)

        # with open(imagefile, "rb") as f:
        #     imagedata = f.read()
        #     base64data = base64.b64encode(imagedata)
        #     base64data = str(base64data, 'utf-8')
        #     item.append(base64data)

    # write to txt
    with open(os.path.join(data_path, custom_data_filename), "w") as f:
        # for item in item_list[:20]:
        for item in item_list:
            f.write("{}\t".format(item[0]))
            f.write("{}\t".format(item[1]))
            f.write("{}\t".format(item[3]))
            f.write("{}\n".format(item[2]))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data_processing")
    parser.add_argument("--data_path", type=str, required=True, help="the path of the original data, including both text and image.")
    parser.add_argument("--custom_data_filename", type=str, required=True, help="the filename of custom data")
    args = parser.parse_args()

    data_process(data_path=args.data_path, custom_data_filename=args.custom_data_filename)

