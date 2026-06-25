import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_constellations():
    print("Lendo o código original da NVIDIA para manter a lógica intacta...")
    with open('nvidia_official.py', 'r', encoding='utf-8') as f:
        original_code = f.read()
    
    class_end_idx = original_code.find("torch._dynamo.reset()")
    if class_end_idx == -1:
        class_end_idx = original_code.find("UL_SIMS = {")
        
    core_logic_code = original_code[:class_end_idx]
    
    local_scope = {}
    exec(core_logic_code, local_scope, local_scope)
    
    Model = local_scope['Model']
    ebnodb2no = local_scope['ebnodb2no']
    cir_to_ofdm_channel = local_scope['cir_to_ofdm_channel']
    
    print("\nInicializando o modelo oficial da NVIDIA...")
    # Inicializando com CDL-C (fading severo)
    model = Model(domain="freq",
                  direction="uplink",
                  cdl_model="C",
                  delay_spread=100e-9,
                  perfect_csi=False,
                  speed=0.0,
                  cyclic_prefix_length=6,
                  pilot_ofdm_symbol_indices=[2, 11])
    
    ebno_db = 15.0 # SNR de 15 dB para visualizar o sinal antes do "pico"
    batch_size = 1 # Apenas 1 batch é necessário para as constelações
    
    print(f"Executando a simulação passo-a-passo (SNR={ebno_db}dB)...")
    
    # 1. Preparar ruído e Transmissão Original
    no = ebnodb2no(ebno_db, model._num_bits_per_symbol, model._coderate, model._rg)
    b = model._binary_source([batch_size, 1, model._num_streams_per_tx, model._k])
    c = model._encoder(b)
    
    # [Tx] Constelação Original
    x = model._mapper(c)
    x_rg = model._rg_mapper(x)
    
    # 2. Canal CDL (Fading Multipercurso)
    cir = model._cdl(batch_size, model._rg.num_ofdm_symbols, 1/model._rg.ofdm_symbol_duration)
    h_freq = cir_to_ofdm_channel(model._frequencies, *cir, normalize=True)
    
    # 3. Receptor Bruto (Y) - Canal + Ruído
    y = model._channel_freq(x_rg, h_freq, no)
    
    # 4. Estimação e Equalização LMMSE
    h_hat, err_var = model._ls_est(y, no)
    x_hat, no_eff = model._lmmse_equ(y, h_hat, err_var, no)
    
    print("Processamento concluído. Extraindo dados para o gráfico...")
    
    # Extração das Constelações para plotagem (convertendo Tensors do PyTorch para Numpy)
    # Sinal Tx
    tx_symbols = x.detach().cpu().numpy().flatten()
    
    # Sinal Rx (Antes da Equalização)
    # Filtramos os subcarriers nulos (zero exato) para não poluir o centro da constelação
    rx_raw = y.detach().cpu().numpy().flatten()
    rx_raw = rx_raw[np.abs(rx_raw) > 1e-6] 
    
    # Sinal Equalizado
    rx_eq = x_hat.detach().cpu().numpy().flatten()
    
    print("Desenhando o Mapa de Estrelas (Constelações)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Grafico 1: Original
    axes[0].scatter(tx_symbols.real, tx_symbols.imag, s=2, color="#2196F3", alpha=0.5)
    axes[0].set_title("1. Sinal Original (Tx)\nConstelação Limpa (QPSK)", fontsize=13)
    
    # Grafico 2: Recebido (Com fading e ruído)
    axes[1].scatter(rx_raw.real, rx_raw.imag, s=2, color="#FF9800", alpha=0.5)
    axes[1].set_title("2. Sinal Recebido (Y = H*X + N)\nTotalmente destorcido pelo Fading (CDL-C)", fontsize=13)
    
    # Grafico 3: Equalizado
    axes[2].scatter(rx_eq.real, rx_eq.imag, s=2, color="#4CAF50", alpha=0.5)
    axes[2].set_title("3. Sinal Equalizado (X_hat = Y / H_hat)\nRecuperado pelo Equalizador LMMSE", fontsize=13)
    
    for ax in axes:
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join("results", "resultado_constelacoes_nvidia.png")
    plt.savefig(output_file, dpi=150)
    print(f"Sucesso! Gráfico salvo em: {output_file}")

if __name__ == '__main__':
    generate_constellations()
