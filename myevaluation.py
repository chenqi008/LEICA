import logging
import os
import sys

import argparse
import base64
from io import BytesIO
from data.file_dataset import FileDataset
from PIL import Image, ImageFile
from torchvision import transforms
from omegaconf import OmegaConf, DictConfig
from models.taming.models.vqgan import GumbelVQ

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm

from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging

from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results

from run_scripts.image_gen import data_processing, generate_code
import evaluate

# for generate_code.py
ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

# for evaluate.py
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


class Eval(object):
	"""docstring for Eval"""
	def __init__(self):
		super(Eval, self).__init__()
		self.args, self.cfg = self.set_config()


	# set configs for all the steps
	def set_config(self):
		# parser = argparse.ArgumentParser(description="data_processing")
		parser = options.get_generation_parser()

		# configs for data_processing.py
		parser.add_argument("--data_path", type=str, required=True, help="the path of the original data, including both text and image.")
		parser.add_argument("--custom_data_filename", type=str, required=True, help="the filename of custom data")
		
		# configs for generate_code.py
		parser.add_argument("--file", type=str, default="")
		parser.add_argument("--outputs", type=str, default="")
		parser.add_argument("--selected_cols", type=str, required=True)
		parser.add_argument("--code_image_size", type=int, required=True)
		parser.add_argument("--vq_model", type=str, required=True)
		parser.add_argument("--vqgan_model_path", type=str, default=None)
		parser.add_argument("--vqgan_config_path", type=str, default=None)
		parser.add_argument("--log_interval", default=100, type=int, help="log interval")
		parser.add_argument("--worker_cnt", type=int, default=1)
		# parser.add_argument("--local_rank", type=int, default=0)
		parser.add_argument("--batch_size", type=int, default=32)

		# the args for step1 and step2
		# args = parser.parse_args()

		# configs for evaluate.py
		# parser2 = options.get_generation_parser()
		parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
		parser.add_argument("--beam-search-vqa-eval", action='store_true', help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
		parser.add_argument("--zero-shot", action='store_true')

		parser.add_argument("--use-indicator", action='store_true', help="Use indication function when calculating OFA score")
		parser.add_argument("--use-credit", action='store_true', help="Use credit assignment when calculating the final score, use patch-wise credit if True")
		parser.add_argument("--use-image-credit", action='store_true', help="Use image-wise credit assignment when calculating the final score")
		parser.add_argument("--use-smooth-exp", action='store_true', help="(Only if --use-image-credit=True) Use smooth exp function in image-wise credit assignment when calculating the final score")

		# two arg files for the step3
		args = options.parse_args_and_arch(parser)
		cfg = convert_namespace_to_omegaconf(args)

		return args, cfg


	# reformat the data that want to be evaluated
	def reformat_data(self):
		# reformat data and generate "custom_data.txt"
		data_processing.data_process(data_path=self.args.data_path, custom_data_filename=self.args.custom_data_filename)


	# generate code according to the pretrained codebook
	def gen_code(self):
		vqgan_config = OmegaConf.load(self.args.vqgan_config_path)
		vqgan = GumbelVQ(**vqgan_config.model.params)
		sd = torch.load(self.args.vqgan_model_path, map_location="cpu")["state_dict"]
		missing, unexpected = vqgan.load_state_dict(sd, strict=False)
		for k, v in vqgan.named_parameters():
			v.requires_grad = False
		image_tokenizer = vqgan.cuda().eval()

		writer = open(self.args.outputs, 'w')

		print("begin process")

		data_cnt = 0

		dataset = generate_code.VQGANDataset(self.args.file, self.args.selected_cols, self.args.code_image_size)
		dataloader = DataLoader(dataset, batch_size=self.args.batch_size)

		for data in tqdm(dataloader):
			batch_size = data["code_image"].size()[0]
			with torch.no_grad():
				z, _, [_, _, image_codes] = image_tokenizer.encode(data["code_image"].cuda())
				image_codes = image_codes.view(batch_size, -1).detach()

			for i, image_code in enumerate(image_codes):
				code = ' '.join([str(num) for num in image_code.tolist()])

				if len(data.keys()) == 4:
					writer.write('\t'.join([data['pair_id'][i], data['image_id'][i], data['text'][i], code])+'\n')
				elif len(data.keys()) == 2:
					writer.write('\t'.join([data['image_id'][i], code])+'\n')
				else:
					raise NotImplementedError
		writer.close()


	# obtain the score from score-based model
	def get_score(self):
		distributed_utils.call_main(self.cfg, evaluate.main_eval, ema_eval=self.args.ema_eval, beam_search_vqa_eval=self.args.beam_search_vqa_eval, zero_shot=self.args.zero_shot, use_indicator=self.args.use_indicator, use_credit=self.args.use_credit, use_image_credit=self.args.use_image_credit, use_smooth_exp=self.args.use_smooth_exp)


if __name__ == '__main__':
	eval_tool = Eval()
	# 
	eval_tool.reformat_data()
	eval_tool.gen_code()
	eval_tool.get_score()
