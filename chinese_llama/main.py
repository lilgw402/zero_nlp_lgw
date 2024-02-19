
import logging
import os
import sys
import json
import click
import numpy as np
from datasets import load_dataset
# import jieba
# from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
from functools import partial

from typing import List, Dict, Optional, Union
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainingArguments
)
from trainer import Trainer
# from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

#加载一个预训练的分词器（Tokenizer）。在自然语言处理中，分词器用于将原始文本字符串分割成更小的单元（通常是词或者子词），这些单元用于模型的输入
def load_tokenizer(model_name_or_path: str = "fastchat/tokenizer"):
    logger.info(f"init tokenizer")
    from fastchat.tokenizer.tokenization_llama_zh import LlamazhTokenizer #LlamazhTokenizer` 类。这个类是 `Llama` 分词器的中国版本
    tokenizer = LlamazhTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)
    return tokenizer

#特定的配置创建 `LlamaForCausalLM` 模型实例，并将其部署到（转至）CUDA设备上。`device_map` 参数指定了多GPU并行。
def load_model(tokenizer, model_name_or_path: Optional[str] = None, device_map: Optional[Dict[int, List[int]]] = None):

    from transformers.models.llama import LlamaConfig
    logger.info("init model")

    config = LlamaConfig(vocab_size=tokenizer.__len__(),
                         hidden_size=2048,
                         intermediate_size=5504,  # 11008,
                         num_hidden_layers=32,  # 32,
                         #  num_attention_heads=32,
                         bos_token_id=tokenizer.bos_token_id,
                         eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id,
                         )

    from fastchat.models.llama.modeling_llama_zh import LlamaForCausalLM
    model = LlamaForCausalLM(config=config).to(torch.bfloat16).cuda()
    model.parallelize(device_map=device_map)

    return model

#递归获取给定目录下所有文件的路径。
def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list

#根据路径加载数据文件并使用 `datasets` 库将它们转换为 `Dataset` 对象。
def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = None):

    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )
    return raw_datasets


def load_tokenizer_and_model():

    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, ],
        1: [7, 8, 9, 10, 11, 12, 13, ],
        2: [14, 15, 16, 17, 18, 19, 20, 21, ],
        3: [22, 23, 24, 25, 26, 27, 28, 29],
        4: [30, 31]
    }
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11]
    }
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, ],
        1: [24, 25, 26, 27, 28, 29, 30, 31],
    }

    tokenizer = load_tokenizer()
    model = load_model(tokenizer=tokenizer, device_map=device_map)

    return tokenizer, model

#主要用于处理用于训练语言模型的文本数据。这个函数从原始的数据集中预处理和编码文本，为模型训练准备必要的输入和标签（labels）
def preprocess_function_(examples: Dict, #包含文本数据的字典，可能是从数据集中抽取出的一批样本
                         tokenizer: AutoTokenizer, #用于将文本转换成模型可以处理的格式。
                         max_source_length: int = 1024,
                         max_target_length: int = 1024,
                         prompt_column: Optional[str] = 'q', #数据中提示（比如问题）所在列的名称。
                         response_column: Optional[str] = 'a',
                         history_column: Optional[str] = None, #历史对话数据所在列的名称。
                         ignore_pad_token_for_loss=-100, #在计算损失时应该忽略的特殊标记的标签，通常设置为 -100，这样在计算损失时就可以忽略填充的token。
                         ):
    max_seq_length = max_source_length + max_target_length

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    # 遍历`examples`中的每个样本。对于每个样本，取出对应于 `prompt_column` 和 `response_column` 的文本内容作为`query`和`answer`
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query, answer = examples[prompt_column][i], examples[response_column][i]

            if history_column is None:
                prompt = query
            else:
                prompt = ""
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(
                        turn_idx, old_query, response)
                prompt += "[Round {}]\n问：{}\n答：".format(
                    len(history), query)

            prompt = prompt


            #使用 tokenizer 将 `prompt` 和 `answer` 文本编码为 token ID 序列
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > max_source_length - 1:
                a_ids = a_ids[: max_source_length - 1]

            if len(b_ids) > max_target_length - 2:
                b_ids = b_ids[: max_target_length - 2]

            #构建完整的输入序列，其中包括必要的特殊 token（如 `bos_token_id` 代表开始符号）。
            input_ids = tokenizer.build_inputs_with_special_tokens(
                a_ids, b_ids)

            #计算上下文的长度 `context_length`，这通常是问题部分的长度。
            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            #在上下文结束后的第一个 token 开始创建标签序列，用于后续的语言模型训练。在上下文处标签设为忽略的值（默认为-100），这样损失函数在计算时会跳过这部分。
            labels = [-100] * context_length + input_ids[mask_position+1:]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100)
                          for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

    return model_inputs #最后返回 `model_inputs` 字典，这些都是经过处理、准备发送到模型进行训练的数据。


def train(*,
          dataset_path: str, #输入数据集的路径
          epochs: int,
          per_device_train_batch_size: int,
          per_device_eval_batch_size: int,
          lr: float,
          seed: int, #随机种子，用于复现结果
          logging_steps: int,
          save_steps: int,
          eval_steps: int,
          test_size: Union[float, int],
          save_total_limit: int,
          local_output_dir: str,
          warmup_steps: int,
          max_source_length: int,
          max_target_length: int,
          gradient_accumulation_steps: int):
    set_seed(seed=seed)
    tokenizer, model = load_tokenizer_and_model()

    dataset = load_dataset_from_path(
        data_path=dataset_path, cache_dir="cache_data")['train']
    #将`preprocess_function`应用于数据集，批处理模式开启，移除原始的`q`和`a`列，在十个进程中并行处理
    preprocess_function = partial(preprocess_function_, tokenizer=tokenizer,
                                  max_source_length=max_source_length,
                                  max_target_length=max_target_length,
                                  prompt_column='q',
                                  response_column='a',
                                  history_column=None,
                                  ignore_pad_token_for_loss=-100
                                  )
    dataset = dataset.map(
        function=preprocess_function,
        batched=True,
        desc="Running tokenizer on train dataset",
        remove_columns=['q', 'a'],
        num_proc=10

    )
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.shuffle(seed=seed)
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    print_dataset_example(split_dataset['train'][0])

    label_pad_token_id = - 100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=False,
        remove_unused_columns=False,
        # local_rank=local_rank,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    logger.info("Instantiating Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logger.info("Training")
    trainer.train() #开始训练模型

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)


@click.command()
@click.option("--dataset-path", type=str, default="data/opendata")
@click.option("--epochs", type=int, default=3)
@click.option("--per-device-train-batch-size", type=int, default=3)
@click.option("--per-device-eval-batch-size", type=int, default=1)
@click.option("--lr", type=float, default=1e-5)
@click.option("--seed", type=int, default=42)
@click.option("--logging-steps", type=int, default=10)
@click.option("--save-steps", type=int, default=1000)
@click.option("--eval-steps", type=int, default=500)
@click.option("--test-size", type=int, default=1000)
@click.option("--save-total-limit", type=int, default=10)
@click.option("--local-output-dir", type=str, default="output/llama_zh001")
@click.option("--warmup-steps", type=int, default=1000)
@click.option("--max-source-length", type=int, default=256)
@click.option("--max-target-length", type=int, default=1024)
@click.option("--gradient-accumulation-steps", type=int, default=8)
def main(**kwargs):
    train(**kwargs)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
