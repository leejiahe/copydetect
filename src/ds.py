import pyrootutils

root = pyrootutils.setup_root(search_from = __file__,
                              indicator = [".git", "pyproject.toml"],
                              pythonpath = True,
                              dotenv = True,
                              )

import os
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from src.models.copydetector_module import CopyDetectorModule

ckpt_path = '/home/stevenlee/copydetect/logs/copydetection/runs/2022-12-09_15-47-05/epoch_001.ckpt'
out_path = '/home/stevenlee/copydetect/logs/copydetection/runs/2022-12-09_15-47-05/out.pt'

def main():
    convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, out_path)

    model = CopyDetectorModule.load_from_checkpoint(out_path)

if __name__ == "__main__":
    main()