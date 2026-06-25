import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 1. Carregamento da Lógica Base da NVIDIA para Integração
# ========================================================================
print("Carregando o modelo do Sionna...")
with open('main/nvidia_official.py', 'r', encoding='utf-8') as f:
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
compute_ber = local_scope['compute_ber']

# ========================================================================
# 2. Definição do Receptor Neural (Neural Post-Equalizer)
# ========================================================================
# Essa rede neural atua após o equalizador linear básico para aprender a
# corrigir distorções não-lineares e ruídos complexos usando Inteligência Artificial.
class NeuralPostEqualizer(nn.Module):
    def __init__(self, num_streams=4):
        super().__init__()
        # Entrada: Real e Imag de num_streams (Ex: 4 streams * 2 = 8 features)
        # Saída: Mesmo tamanho, sinal limpo.
        self.net = nn.Sequential(
            nn.Linear(num_streams * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_streams * 2)
        )
        
    def forward(self, x_hat):
        # x_hat shape: [batch_size, 1, num_streams, num_data_symbols]
        batch_size = x_hat.shape[0]
        num_streams = x_hat.shape[2]
        num_symbols = x_hat.shape[-1]
        
        # Remove a dimensão rx (1) e rearranja
        x_h = x_hat.squeeze(1).permute(0, 2, 1) # [batch_size, num_symbols, num_streams]
        
        # Separa Real e Imag
        x_real = torch.cat([x_h.real, x_h.imag], dim=-1).float()
        
        # Passa pela IA
        out = self.net(x_real)
        
        # Junta Real e Imag novamente
        out_real, out_imag = torch.split(out, num_streams, dim=-1)
        out_complex = torch.complex(out_real, out_imag)
        
        # Retorna ao formato original
        return out_complex.permute(0, 2, 1).unsqueeze(1)

# ========================================================================
# 3. Pipeline de Treinamento e Avaliação
# ========================================================================
def run_neural_receiver_demo():
    print("Inicializando o Ambiente de Simulação (CDL-C, SNR=10dB)...")
    model = Model(domain="freq",
                  direction="uplink",
                  cdl_model="C",
                  delay_spread=100e-9,
                  perfect_csi=False,
                  speed=0.0,
                  cyclic_prefix_length=6,
                  pilot_ofdm_symbol_indices=[2, 11])
    
    ebno_db = 10.0
    batch_size = 128
    
    no = ebnodb2no(ebno_db, model._num_bits_per_symbol, model._coderate, model._rg)
    
    neural_eq = NeuralPostEqualizer()
    optimizer = optim.Adam(neural_eq.parameters(), lr=0.005)
    mse_loss = nn.MSELoss()
    
    epochs = 50
    print(f"\nIniciando Treinamento da Rede Neural por {epochs} épocas...")
    
    loss_history = []
    
    # Loop de Treinamento
    for epoch in range(epochs):
        # 1. Gerar Dados (Transmissão)
        b = model._binary_source([batch_size, 1, model._num_streams_per_tx, model._k])
        c = model._encoder(b)
        x = model._mapper(c) # Sinal Original Perfeito X_k
        x_rg = model._rg_mapper(x)
        
        # 2. Canal Fading
        cir = model._cdl(batch_size, model._rg.num_ofdm_symbols, 1/model._rg.ofdm_symbol_duration)
        h_freq = cir_to_ofdm_channel(model._frequencies, *cir, normalize=True)
        
        # 3. Sinal Recebido e Equalização Linear (Matemática Tradicional)
        y = model._channel_freq(x_rg, h_freq, no)
        h_hat, err_var = model._ls_est(y, no)
        x_hat_linear, no_eff = model._lmmse_equ(y, h_hat, err_var, no)
        
        # 4. Equalização Neural (Inteligência Artificial)
        optimizer.zero_grad()
        x_hat_neural = neural_eq(x_hat_linear)
        
        # 5. Calcular o Erro (Diferença entre a saída da IA e o X real)
        # Comparar no domínio real e imaginário
        loss = mse_loss(torch.cat([x_hat_neural.real, x_hat_neural.imag], dim=-1),
                        torch.cat([x.real, x.imag], dim=-1))
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Época {epoch+1}/{epochs} | Loss (Erro): {loss.item():.4f}")

    print("\nTreinamento Concluído! Gerando gráficos comparativos...")
    
    # Avaliação Visual (Constelações)
    # Pegamos apenas 1 batch para plotar a constelação
    x_linear_np = x_hat_linear.detach().cpu().numpy().flatten()
    x_neural_np = x_hat_neural.detach().cpu().numpy().flatten()
    x_orig_np = x.detach().cpu().numpy().flatten()
    
    os.makedirs('extras/results', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].scatter(x_linear_np.real, x_linear_np.imag, s=1, alpha=0.5, color='orange')
    axes[0].set_title("Equalizador LMMSE (Tradicional)", fontsize=12)
    
    axes[1].scatter(x_neural_np.real, x_neural_np.imag, s=1, alpha=0.5, color='green')
    axes[1].set_title("Receptor Neural (Inteligência Artificial)", fontsize=12)
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    output_path = os.path.join('extras', 'results', 'comparacao_neural_receiver.png')
    plt.savefig(output_path, dpi=150)
    print(f"Gráfico de constelação salvo em: {output_path}")

    # Gráfico de Convergência da Loss
    plt.figure()
    plt.plot(loss_history, linewidth=2, color='purple')
    plt.title("Evolução do Aprendizado da Rede Neural")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Médio Quadrático (MSE)")
    plt.grid(True)
    loss_path = os.path.join('extras', 'results', 'treinamento_neural.png')
    plt.savefig(loss_path, dpi=150)
    print(f"Gráfico de erro salvo em: {loss_path}")

if __name__ == '__main__':
    run_neural_receiver_demo()
