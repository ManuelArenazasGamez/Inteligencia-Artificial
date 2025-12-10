import tkinter as tk
from tkinter import scrolledtext, messagebox
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import threading

# --- CONFIGURACIÓN ---
NOMBRE_MODELO_BASE = "meta-llama/Llama-3.2-3B-Instruct" 
CARPETA_ADAPTADORES = "./lora-tutor"

class TutorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto 4: Tutor Inteligente de Algoritmos")
        self.root.geometry("600x500")
        
        # 1. Área de Chat 
        self.chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.tag_config('usuario', foreground='blue', font=("Arial", 10, "bold"))
        self.chat_area.tag_config('tutor', foreground='green')
        self.chat_area.tag_config('sistema', foreground='gray', font=("Arial", 8, "italic"))

        # 2. Área de Entrada 
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.entry_msg = tk.Entry(self.input_frame, font=("Arial", 12))
        self.entry_msg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry_msg.bind("<Return>", self.enviar_mensaje) # Enviar con Enter
        
        self.btn_enviar = tk.Button(self.input_frame, text="Preguntar", command=self.enviar_mensaje, bg="#DDDDDD")
        self.btn_enviar.pack(side=tk.RIGHT)

        # 3. Cargar el Modelo 
        self.mostrar_en_chat("SISTEMA", "Cargando modelo... esto puede tardar unos segundos.\n")
        self.root.update() # Forzar actualización visual
        
        # Iniciar carga
        threading.Thread(target=self.cargar_modelo, daemon=True).start()

    def cargar_modelo(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO_BASE)
            
            # Carga base
            base_model = AutoModelForCausalLM.from_pretrained(
                NOMBRE_MODELO_BASE,
                load_in_8bit=True, 
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Carga tus cambios (Fine-Tuning)
            self.model = PeftModel.from_pretrained(base_model, CARPETA_ADAPTADORES)
            self.model.eval()
            
            self.mostrar_en_chat("SISTEMA", "¡Modelo cargado exitosamente! Puedes empezar.\n")
        except Exception as e:
            self.mostrar_en_chat("ERROR", f"No se pudo cargar el modelo: {str(e)}")

    def enviar_mensaje(self, event=None):
        pregunta = self.entry_msg.get()
        if not pregunta.strip():
            return
            
        self.mostrar_en_chat("TÚ", pregunta + "\n")
        self.entry_msg.delete(0, tk.END)
        self.btn_enviar.config(state=tk.DISABLED) 
        
        threading.Thread(target=self.generar_respuesta, args=(pregunta,)).start()

    def generar_respuesta(self, pregunta):
        try:
            prompt = f"Instrucción: {pregunta}\nRespuesta:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True
                )
            
            respuesta_completa = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Limpiar para mostrar solo la respuesta
            if "Respuesta:" in respuesta_completa:
                respuesta_final = respuesta_completa.split("Respuesta:")[-1].strip()
            else:
                respuesta_final = respuesta_completa

            self.mostrar_en_chat("TUTOR", respuesta_final + "\n")
            
        except Exception as e:
            self.mostrar_en_chat("ERROR", f"Falló la generación: {str(e)}")
        finally:
            self.root.after(0, lambda: self.btn_enviar.config(state=tk.NORMAL))

    def mostrar_en_chat(self, emisor, texto):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"{emisor}: ", emisor.lower()) # Etiqueta para color
        self.chat_area.insert(tk.END, texto + "\n")
        self.chat_area.see(tk.END) # Auto-scroll al final
        self.chat_area.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TutorApp(root)
    root.mainloop()