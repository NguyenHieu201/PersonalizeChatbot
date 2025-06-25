from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


def format_chat_template(row):
    # row_json = [{"role": "system", "content": row["instruction"]},
    #            {"role": "user", "content": row["input"]},
    #            {"role": "assistant", "content": row["output"]}]
    # row_json = [
    #     {"role": "user", "content": row["Prompt"]},
    #     {"role": "assistant", "content": row["Response"]}
    # ]
    # row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    
    row["text"] = f"<start_of_turn>user\n{row['Prompt']}\n<end_of_turn>\n<start_of_turn>model\n{row['Response']}\n<end_of_turn>"
    return row


if __name__ == "__main__":
    from huggingface_hub import login
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()

    hf_token = user_secrets.get_secret("hf-token")
    login(token = hf_token)

    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    dataset_name = "TuringsSolutions/BatmanAI" # Choose any dataset! We support all Hugging Face datasets.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
        "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/Qwen2-0.5b-bnb-4bit",           # Qwen2 2x faster!
        "unsloth/Qwen2-1.5b-bnb-4bit",
        "unsloth/Qwen2-7b-bnb-4bit",
        "unsloth/Qwen2-72b-bnb-4bit",
        "unsloth/gemma-2-9b-bnb-4bit",           # 8T tokens 2x faster!
        "unsloth/gemma-2-27b-bnb-4bit",          # 13T tokens 2x faster!
    ] # Try more models at https://huggingface.co/unsloth!

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2-9b", # Reminder we support ANY Hugging Face model!
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=65).select(range(906))
    dataset = dataset.map(
        format_chat_template,
        num_proc=4
    )
    dataset = dataset.train_test_split(test_size=0.1)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps = 5,
            max_steps = 60,

            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )
    
    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained("lora_model") # Local saving
    tokenizer.save_pretrained("lora_model")