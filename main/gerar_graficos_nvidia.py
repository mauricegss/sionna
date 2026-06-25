import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_and_run_quick_sim():
    print("Lendo o código original da NVIDIA para manter a lógica intacta...")
    with open('nvidia_official.py', 'r', encoding='utf-8') as f:
        original_code = f.read()
    
    # Vamos extrair apenas as importações e a classe Model do código original, 
    # para não rodar a simulação pesada de 2 horas da NVIDIA no escopo global.
    # A classe Model vai até o final da declaração da sua função call, que termina logo antes de "torch._dynamo.reset()"
    
    class_end_idx = original_code.find("torch._dynamo.reset()")
    if class_end_idx == -1:
        # Se não encontrar, pega tudo até onde achar UL_SIMS
        class_end_idx = original_code.find("UL_SIMS = {")
        
    core_logic_code = original_code[:class_end_idx]
    
    # Executamos o código base no nosso escopo para termos a classe Model disponível
    # Isso garante que estamos usando exatamente a matemática da NVIDIA!
    local_scope = {}
    exec(core_logic_code, local_scope, local_scope)
    
    Model = local_scope['Model']
    sim_ber = local_scope['sim_ber']
    
    print("\nExecutando simulação rápida (Apenas CDL-C e SNR até 16dB) para gerar os gráficos...")
    
    # Simulação Rápida (Apenas um modelo de canal para não demorar horas)
    cdl_model = "C"
    ebno_db = list(np.arange(0, 17, 2.0)) # SNR de 0 a 16 dB
    
    model = Model(domain="freq",
                  direction="uplink",
                  cdl_model=cdl_model,
                  delay_spread=100e-9,
                  perfect_csi=False, # CSI Imperfeito (Faz a estimação de canal usando Pilotos + LMMSE!)
                  speed=0.0,
                  cyclic_prefix_length=6,
                  pilot_ofdm_symbol_indices=[2, 11])
    
    # Aqui usamos sim_ber da NVIDIA. 
    # Para ser mais rápido, limitamos max_mc_iter = 100
    start = time.time()
    ber, bler = sim_ber(model,
                        ebno_db,
                        batch_size=128,          # Quantidade de simulações por lote
                        max_mc_iter=50,         # Vários lotes para Monte Carlo evitar os "picos" aleatórios
                        num_target_block_errors=100, 
                        target_bler=1e-3)
    duration = time.time() - start
    print(f"Simulação concluída em {duration/60:.2f} minutos.")
    
    # Gerar Gráfico
    plt.figure(figsize=(8,6))
    plt.semilogy(ebno_db, ber.cpu().numpy(), 'bo-', linewidth=2, label=f"NVIDIA CDL-{cdl_model} (Estimação LS + LMMSE)")
    
    plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    plt.ylabel("BER (Bit Error Rate)", fontsize=12)
    plt.grid(which="both", linestyle="--", alpha=0.5)
    plt.title("Validação da Lógica NVIDIA: Curva BER vs SNR", fontsize=14)
    plt.ylim([1e-5, 1.0])
    plt.legend()
    
    output_img = os.path.join("results", "resultado_nvidia_ber.png")
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {output_img}")
    
if __name__ == '__main__':
    extract_and_run_quick_sim()
