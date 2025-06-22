import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import glob


def main():
    base_model_path = "./model/Qwen3-8B"
    lora_adapter_path = "./output/Qwen3_8B_lora/checkpoint-750"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, model_id=lora_adapter_path)
    model.to(device)

    test_files = glob.glob("./dataset/test*.json")
    test_data = []
    for test_file in test_files:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data.extend(json.load(f))

    with open("results.txt", "w", encoding="utf-8") as f_out:
        for item in test_data:
            prompt = item["content"]
            print("Generating:", prompt)

            inputs = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个四元组。请从下面的文本抽取一个或多个四元组，每一个四元组输出格式为评论对象|对象观点|是否仇恨|仇恨群体。评论对象可以为'NULL',对象观点尽量简洁,仇恨群体只包括(LGBTQ、Region、Sexism、Racism、others、non-hate)，同一四元组可能涉及多个仇恨群体，是否仇恨标签为(hate、non-hate),多个四元组之间用[SEP]分隔,最后一个四元组后面加[END]。提取出句子中包含的所有四元组。",
                    },
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ).to(device)

            gen_kwargs = {"max_length": 1000, "do_sample": True, "top_k": 1}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("Result: ", result)
                f_out.write(result + "\n")


if __name__ == "__main__":
    main()
