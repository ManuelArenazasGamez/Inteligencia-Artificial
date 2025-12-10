import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- CONFIGURACIÓN ---
ruta_modelo_base = "microsoft/Phi-3-mini-4k-instruct"
ruta_adaptadores = "./tutor_ajustado_phi3" # TU CARPETA GENERADA

print(f"--- Cargando Modelo Base: {ruta_modelo_base} ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    ruta_modelo_base,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo_base, trust_remote_code=True)

print(f"--- Cargando TUS conocimientos (LoRA): {ruta_adaptadores} ---")
# AQUÍ SE FUSIONA TU ENTRENAMIENTO CON EL MODELO
model = PeftModel.from_pretrained(model, ruta_adaptadores)

print("\n" + "="*50)
print(" ¡TUTOR INTELIGENTE ACTIVO! (Escribe 'salir' para terminar)")
print("="*50 + "\n")

def generar_respuesta(pregunta):
    instrucciones = "Eres un Tutor de Algoritmos experto. Explica el algoritmo paso a paso y usa una analogía simple."
    
    prompt = f"<|user|>\n{instrucciones}\n\nPregunta: {pregunta}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    input_length = inputs.input_ids.shape[1] 
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.2,     
            do_sample=True,
            use_cache=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    

    generated_tokens = outputs[0][input_length:]
    
    respuesta_limpia = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return respuesta_limpia.strip()

# Bucle de chat
while True:
    usuario = input("\nEstudiante: ")
    if usuario.lower() in ["salir", "exit"]:
        break
    
    print("Tutor: Pensando...", end="\r")
    respuesta = generar_respuesta(usuario)
    print(f"Tutor: {respuesta}")