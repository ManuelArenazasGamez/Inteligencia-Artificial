import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

if __name__ == "__main__":
    
    # 1. Configuración
    archivo_dataset = "tutor_programacion.jsonl" 
    output_dir = "./tutor_ajustado_phi3"
    model_name = "microsoft/Phi-3-mini-4k-instruct" 

    print(f"--- Cargando dataset desde {archivo_dataset} ---")
    dataset = load_dataset("json", data_files=archivo_dataset)

    # 2. Tokenizer
    print("--- Cargando Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Modelo Base (Configurado para Windows + 8bit)
    print("--- Cargando Modelo en 8-bits ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  
        device_map="auto",
        trust_remote_code=True
    )

    # PASO CRÍTICO: Preparar modelo para entrenamiento k-bit
 
    model = prepare_model_for_kbit_training(model)

    # 4. Configurar LoRA
    print("--- Configurando LoRA ---")
    lora_config = LoraConfig(
        r=16, # Rango de atención
        lora_alpha=32,
        target_modules="all-linear", # Apunta a todas las capas lineales (mejor aprendizaje)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    # 5. Preprocesar datos
    def format_instruction(example):
       
        pregunta = example.get('prompt', example.get('instruction', ''))
        respuesta = example.get('response', '')
        
        # Formato ChatML para Phi-3
        text = f"<|user|>\n{pregunta}<|end|>\n<|assistant|>\n{respuesta}<|end|>"
        return tokenizer(text, truncation=True, max_length=512)

    print("--- Tokenizando el dataset ---")
    tokenized = dataset.map(format_instruction)

    # 6. Entrenamiento
    print("--- Configurando Argumentos de Entrenamiento ---")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Mantenlo en 1 para Windows
        gradient_accumulation_steps=4,
        logging_steps=1,
        num_train_epochs=15, #   épocas requeridas
        learning_rate=2e-3, # learning rate requerido
        fp16=True, 
        save_strategy="epoch",
        optim="adamw_8bit", # Optimizador compatible con bitsandbytes
        report_to="none" # Desactiva wandb para que no pida login
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("--- INICIANDO FINE-TUNING REAL ---")
    trainer.train()

    # 7. Guardar
    print("--- Guardando Adaptadores ---")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"¡PROCESO TERMINADO! Revisa la carpeta {output_dir}")