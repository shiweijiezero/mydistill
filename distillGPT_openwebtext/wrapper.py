import logging
import time

import psutil
import pytorch_lightning as pl
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import nn
from torch.optim import AdamW
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    # get_linear_schedule_with_warmup,
)

import util

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


class Wrapper(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        self.minibatch_size = config["batch_size"]
        # self.automatic_optimization = False # 手动优化
        self.train_dataset_size = config["train_data_size"]
        self.config = config
        self.ts = time.time()
        self.counter = 0
        self.myutil = util.UtilClass()

        # TOKENIZER #
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_model_input_size = self.tokenizer.max_model_input_sizes["gpt2"]

        # TEACHER #
        self.teacher = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
        logger.info(f"Teacher loaded from gpt2.")
        logger.info(f"self.teacher.config {self.teacher.config}")

        # EXTRACT PARAMETERS(初始化参数)
        dump_checkpoint = self.myutil.extract_parameters(self.teacher)

        # STUDENT
        stu_architecture_config = GPT2Config.from_pretrained("stu.json")
        stu_architecture_config.output_hidden_states = True

        self.student = GPT2LMHeadModel.from_pretrained(dump_checkpoint, config=stu_architecture_config)
        logger.info("Student loaded.")

        # FREEZING #
        self.freeze_pos_embeddings(self.student)
        self.freeze_param(self.teacher)

        # SANITY CHECKS #
        assert self.student.config.vocab_size == self.teacher.config.vocab_size
        assert self.student.config.hidden_size == self.teacher.config.hidden_size
        assert self.student.config.max_position_embeddings == self.teacher.config.max_position_embeddings

        # 加载训练参数
        self.vocab_size = self.student.config.vocab_size

        self.temperature = config["temperature"]
        assert self.temperature > 0.0

        self.alpha_ce = config["alpha_ce"]  # 交叉熵损失缩放稀疏
        self.alpha_mlm = config["alpha_mlm"]
        self.alpha_clm = config["alpha_clm"]
        self.alpha_mse = config["alpha_mse"]
        self.alpha_cos = config["alpha_cos"]
        logger.info("Using CLM loss for LM step.")

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
        self.save_step_num = 0

    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset_size = self.train_dataset_size
        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.minibatch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def freeze_pos_embeddings(self, student):
        student.transformer.wpe.weight.requires_grad = False

    def freeze_param(self,teacher):
        for param in teacher.parameters():
            param.requires_grad = False

    # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
    def training_step(self, input_, batch_idx):
        """
        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        if(batch_idx == 0):
            print(input_[0].shape)
            print(input_[1].shape)
            print(input_[2].shape)



        input_ids, attention_mask, lm_labels = input_

        self.ts = time.time()

        student_outputs = self.student(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=None)  # (bs, seq_length, voc_size)
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        assert s_logits.size() == t_logits.size()

        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
        mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)，扩充复制最后一维
        # Mask 学生与教师输出
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        # 计算loss
        # 交叉熵用的KL散度来实现
        loss_ce = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                    nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
                )
                * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
            loss += self.alpha_mlm * loss_mlm

        if self.alpha_clm > 0.0:
            # Shift so that tokens < n predict n,语言模型要求用前一个预测下一个，因此label需要整体向后移一个位置
            shift_logits = s_logits[..., :-1, :].contiguous()  # 丢弃seq的最后一个
            shift_labels = lm_labels[..., 1:].contiguous()  # 丢弃seq的第一个
            # Flatten the tokens
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse

        if self.alpha_cos > 0.0:
            # 使用cosine loss计算最后一层输出的相似度，优化目标为cosine值为1,尽可能保证两个输出方向一致
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos

        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        # lr = self.scheduler.get_last_lr()[0]
        lr = self.optimizer.param_groups[0]['lr']
        log_dic = {
            'loss/loss': self.last_loss,
            'loss/last_loss_ce': self.last_loss_ce,
            'loss/last_loss_clm': self.last_loss_clm,
            # 'last_loss_cos': self.last_loss_cos,
            "loss/lr": lr
        }
        # if(batch_idx%5==0):
        #     print(f"self.lr:{lr}")
            # print(f"self.optimizer:{self.optimizer.param_groups[0]['lr']}")
        for param_name, param in self.student.named_parameters():
            log_dic["parameter_mean/" + param_name] = param.data.mean()
            log_dic["parameter_std/" + param_name] = param.data.std()
            if param.grad is None:
                continue
            log_dic["grad_mean/" + param_name] = param.grad.data.mean()
            log_dic["grad_std/" + param_name] = param.grad.data.std()
        log_dic["usage/memory_usage"] = psutil.virtual_memory()._asdict()["used"] / 1_000_000 / 1_000_000
        log_dic["usage/time_per_step"] = time.time() - self.ts

        self.log_dict(log_dic, prog_bar=True,logger=True)

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        print(f"--- Saving model:./save_model/gpt2_hf_{self.save_step_num}_{self.last_loss}")
        self.student.save_pretrained(f"./save_model/gpt2_{self.save_step_num}_{self.last_loss}", push_to_hub=False)
        self.tokenizer.save_pretrained(f"./save_model/gpt2_{self.save_step_num}_{self.last_loss}", push_to_hub=False)
        self.save_step_num += 1

    def on_train_end(self) -> None:
        print(f"--- Saving model:./save_model/gpt2_hf_end_{self.last_loss}")
        self.student.save_pretrained(f"./save_model/gpt2_hf_end", push_to_hub=False)
        self.tokenizer.save_pretrained(f"./save_model/gpt2_hf_end", push_to_hub=False)


    # def forward(self, input_image):
    #     ourput_image = self.model(input_image)
    #     return ourput_image



    def configure_optimizers(self):
        logger.info("--- Initializing model optimizer")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.student.named_parameters() if
                    not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [
                    p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        logger.info(f"learning_rate: {self.config['learning_rate']}")
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.config["learning_rate"], eps=self.config["adam_epsilon"],
            betas=(0.9, 0.98)
        )

        # warmup_steps = math.ceil(self.num_training_steps * self.config["warmup_prop"])
        warmup_steps = 100
        cycle_steps=2000


        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        # print(f"warmup_steps:{warmup_steps}")
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cycle_steps,
            cycle_mult=1.0,
            max_lr=self.config["learning_rate"],
            min_lr=0,
            warmup_steps=warmup_steps
        )

        print(f"self.num_training_steps:{self.num_training_steps}")

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler
        }
