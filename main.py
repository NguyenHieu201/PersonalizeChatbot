from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
import bitsandbytes as bnb


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def format_chat_template(row):
    # row_json = [{"role": "system", "content": row["instruction"]},
    #            {"role": "user", "content": row["input"]},
    #            {"role": "assistant", "content": row["output"]}]
    row_json = [
        {"role": "user", "content": row["Prompt"]},
        {"role": "assistant", "content": row["Response"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

if __name__ == "__main__":
    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()

    hf_token = user_secrets.get_secret("hf-token")
    login(token = hf_token)

    base_model = "google/gemma-2-9b-it"
    dataset_name = "TuringsSolutions/BatmanAI"
    new_model = "Gemma-2-9b-batman-ai"
    
    # if torch.cuda.get_device_capability()[0] >= 8:
    #     torch_dtype = torch.bfloat16
    #     attn_implementation = "flash_attention_2"
    # else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    modules = find_all_linear_names(model)
    # LoRA config
    peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules
    )
    try:
        model, tokenizer = setup_chat_format(model, tokenizer)
    except ValueError:
        pass
    model = get_peft_model(model, peft_config)

    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=65).select(range(906))
    dataset = dataset.map(
        format_chat_template,
        num_proc=4
    )
    dataset = dataset.train_test_split(test_size=0.1)

    # Setting Hyperparamter
    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        group_by_length=True,
        report_to="wandb",
    )
    # Setting sft parameters
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     peft_config=peft_config,
    #     # max_seq_length= 512,
    #     # dataset_text_field="text",
    #     # tokenizer=tokenizer,
    #     args=training_arguments,
    #     # packing= False,
    #     data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    # )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=TrainingArguments(
            per_device_train_batch_size=8,
            auto_find_batch_size=True,
            gradient_accumulation_steps=4,
            warmup_steps=2,
    #         num_train_epochs=Config.train_epoch,
            max_steps=100,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            eval_strategy="steps",   # "epoch"
            save_steps=5,
            save_total_limit=2,
            load_best_model_at_end=True
        ),
        peft_config=peft_config
    )

    model.config.use_cache = False
    trainer.train()
    trainer.model.save_pretrained(new_model)