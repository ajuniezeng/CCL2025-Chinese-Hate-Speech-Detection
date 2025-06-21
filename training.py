from datasets import Dataset
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from peft import LoraConfig, TaskType, get_peft_model
import functools


def process_func(example, tokenizer):
    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<s><|im_start|>system\n你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个四元组。请从下面的文本抽取一个或多个四元组，每一个四元组输出格式为评论对象|对象观点|是否仇恨|仇恨群体。评论对象可以为'NULL',对象观点尽量简洁,仇恨群体只包括(LGBTQ、Region、Sexism、Racism、others、non-hate)，同一四元组可能涉及多个仇恨群体，是否仇恨标签为(hate、non-hate),多个四元组之间用[SEP]分隔,最后一个四元组后面加[END]。提取出句子中包含的所有四元组:<|im_end|>\n"
        f"<|im_start|>user\n{example['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    df = pd.read_json("./dataset/train.json")
    ds = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(
        "./model", use_fast=False, trust_remote_code=True
    )
    tokenized_id = ds.map(
        functools.partial(process_func, tokenizer=tokenizer),
        remove_columns=ds.column_names,
    )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./output/Qwen3_8B_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
