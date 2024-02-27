import os
from pprint import pprint

import torch


class UtilClass():
    def __init__(self):
        pass

    def extract_parameters(self, teacher_model):
        model = teacher_model
        dump_checkpoint = "./extract_para/pytorch_model.bin"
        return_dump = "./extract_para"
        if (os.path.exists(dump_checkpoint)):
            return return_dump
        if(not os.path.exists(return_dump)):
            os.mkdir(return_dump)

        prefix = "transformer"

        state_dict = model.state_dict()
        pprint(list(state_dict.keys()))
        compressed_sd = {}

        # Embeddings #
        for param_name in ["wte.weight", "wpe.weight"]:
            compressed_sd[f"{prefix}.{param_name}"] = state_dict[f"{prefix}.{param_name}"]

        # Transformer Blocks #
        # 将教师模型对0,2,4,7,9,11层的参数进行赋予学生模型，进行初始化
        std_idx = 0
        for teacher_idx in [0, 2, 4, 7, 9, 11]:
            for layer in ["ln_1", "attn.c_attn", "attn.c_proj", "ln_2", "mlp.c_fc", "mlp.c_proj"]:
                for w in ["weight", "bias"]:
                    compressed_sd[f"{prefix}.h.{std_idx}.{layer}.{w}"] = state_dict[
                        f"{prefix}.h.{teacher_idx}.{layer}.{w}"
                    ]
            # compressed_sd[f"{prefix}.h.{std_idx}.attn.bias"] = state_dict[f"{prefix}.h.{teacher_idx}.attn.bias"]
            std_idx += 1

        # Language Modeling Head ###s
        for w in ["weight", "bias"]:
            compressed_sd[f"{prefix}.ln_f.{w}"] = state_dict[f"{prefix}.ln_f.{w}"]
        compressed_sd["lm_head.weight"] = state_dict["lm_head.weight"]

        print(f"N layers selected for distillation: {std_idx}")
        print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")

        print(f"Save transferred checkpoint to {dump_checkpoint}.")
        torch.save(compressed_sd, dump_checkpoint)
        return return_dump


if __name__ == "__main__":
    myutils = UtilClass()
