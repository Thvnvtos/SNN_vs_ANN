import os, torch, random
import numpy as np

def check_dirs(logs_path, save_path):
	
	logs_path = os.path.join("logs",logs_path)
	save_path = os.path.join("saved_models",save_path)

	if not os.path.exists(logs_path):
		if not os.path.exists("logs"): os.mkdir("logs")
		os.mkdir(logs_path)

	if not os.path.exists(save_path):
		if not os.path.exists("saved_models"): os.mkdir("saved_models")
		os.mkdir(save_path)
		
	return logs_path, save_path

def set_seed(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True