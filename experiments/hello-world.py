import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# ===========================================================================
# Imports do Sionna PHY 2.0
# ===========================================================================
from sionna.phy.mapping import (
    BinarySource,
    Constellation,
    Mapper,
    Demapper,
)
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, compute_ber

# ===========================================================================
# 1. PARÂMETROS DA SIMULAÇÃO
# ===========================================================================
BATCH_SIZE = 1024          # Número de blocos por iteração
K = 512                    # Bits de informação por palavra-código
N = 1024                   # Comprimento da palavra-código (taxa = K/N = 0.5)
NUM_BITS_PER_SYMBOL = 4    # 16-QAM (2^4 = 16 pontos na constelação)
CODERATE = K / N           # Taxa de código = 0.5

# Faixa de Eb/N₀ para varredura (dB)
EBN0_DB_MIN = -2.0
EBN0_DB_MAX = 10.0
EBN0_DB_STEP = 0.5
EBN0_DB_RANGE = np.arange(EBN0_DB_MIN, EBN0_DB_MAX + EBN0_DB_STEP, EBN0_DB_STEP)

# Diretório para salvar gráficos
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print(" SIONNA PHY -- Hello World: LDPC + 16-QAM sobre AWGN")
print("=" * 65)
print(f" Sionna backend: PyTorch")
print(f" Batch size:     {BATCH_SIZE}")
print(f" Info bits (k):  {K}")
print(f" Code len (n):   {N}")
print(f" Code rate:      {CODERATE:.2f}")
print(f" Modulacao:      {2**NUM_BITS_PER_SYMBOL}-QAM")
print(f" Eb/N0 range:    {EBN0_DB_MIN} a {EBN0_DB_MAX} dB")
print("=" * 65)

# ===========================================================================
# 2. INSTANCIAÇÃO DOS COMPONENTES (CAMADAS)
# ===========================================================================
print("\n[1/4] Instanciando componentes do sistema...")


# Fonte de bits aleatórios
source = BinarySource()

# Codificador e decodificador LDPC 5G NR
encoder = LDPC5GEncoder(k=K, n=N)
decoder = LDPC5GDecoder(encoder=encoder)

# Constelação e mapeador/demapeador QAM
constellation = Constellation("qam", num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# Canal AWGN
channel = AWGN()

print("   [OK] BinarySource")
print("   [OK] LDPC5GEncoder / LDPC5GDecoder")
print(f"   [OK] Constellation ({2**NUM_BITS_PER_SYMBOL}-QAM)")
print("   [OK] Mapper / Demapper (APP)")
print("   [OK] Canal AWGN")

# ===========================================================================
# 3. SIMULAÇÃO: VARREDURA BER x Eb/N0
# ===========================================================================
print("\n[2/4] Executando simulacao BER x Eb/N0...")

ber_results = []
snr_simulated = []

for i, ebno_db in enumerate(EBN0_DB_RANGE):
    # --- Transmissor ---
    # Gera bits aleatórios
    bits = source([BATCH_SIZE, K])

    # Codifica com LDPC 5G
    codewords = encoder(bits)

    # Mapeia para símbolos QAM
    symbols = mapper(codewords)

    # --- Canal ---
    # Converte Eb/N0 (dB) para variancia do ruido (No)
    no = ebnodb2no(
        ebno_db=ebno_db,
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        coderate=CODERATE,
    )
    # Aplica ruído AWGN
    noisy_symbols = channel(symbols, no)

    # --- Receptor ---
    # Demapeia: calcula LLRs (Log-Likelihood Ratios)
    llr = demapper(noisy_symbols, no)

    # Decodifica LDPC
    bits_hat = decoder(llr)

    # --- Métrica ---
    ber = compute_ber(bits, bits_hat)
    ber_val = ber.numpy() if hasattr(ber, 'numpy') else float(ber)
    ber_results.append(ber_val)
    snr_simulated.append(ebno_db)

    # Progresso
    bar_len = 30
    progress = (i + 1) / len(EBN0_DB_RANGE)
    bar = "#" * int(bar_len * progress) + "." * (bar_len - int(bar_len * progress))
    print(f"\r   [{bar}] {progress*100:5.1f}% | Eb/N0={ebno_db:+5.1f} dB | BER={ber_val:.2e}", end="")

print()  # Nova linha após barra de progresso

ber_results = np.array(ber_results)
snr_simulated = np.array(snr_simulated)

# ===========================================================================
# 4. GRÁFICO 1: CURVA BER x Eb/N0
# ===========================================================================
print("\n[3/4] Gerando gráficos...")

fig, ax = plt.subplots(figsize=(10, 7))

# Curva simulada (clip para evitar log(0))
ax.semilogy(
    snr_simulated,
    np.clip(ber_results, 1e-7, 1),
    "bo-",
    linewidth=2,
    markersize=5,
    label=f"Simulado: LDPC({N},{K}) + {2**NUM_BITS_PER_SYMBOL}-QAM",
)

# Curva teorica QAM nao-codificada (referencia)
from scipy.special import erfc
ebno_lin = 10 ** (EBN0_DB_RANGE / 10)
M = 2 ** NUM_BITS_PER_SYMBOL
ber_uncoded = (2 * (np.sqrt(M) - 1) / (np.sqrt(M) * NUM_BITS_PER_SYMBOL)) * \
              erfc(np.sqrt(3 * NUM_BITS_PER_SYMBOL * ebno_lin / (2 * (M - 1))))
ax.semilogy(
    EBN0_DB_RANGE,
    np.clip(ber_uncoded, 1e-7, 1),
    "r--",
    linewidth=1.5,
    label=f"Teórico: {2**NUM_BITS_PER_SYMBOL}-QAM não-codificada",
)

ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=13)
ax.set_ylabel("BER (Bit Error Rate)", fontsize=13)
ax.set_title("Desempenho BER - Sistema LDPC + QAM sobre Canal AWGN\n(Sionna PHY 2.0)", fontsize=14)
ax.set_xlim([EBN0_DB_MIN, EBN0_DB_MAX])
ax.set_ylim([1e-6, 1])
ax.grid(True, which="both", alpha=0.3, linestyle="--")
ax.legend(fontsize=11, loc="lower left")
ax.tick_params(labelsize=11)

plt.tight_layout()
ber_plot_path = os.path.join(OUTPUT_DIR, "ber_curve.png")
plt.savefig(ber_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Curva BER salva em: {ber_plot_path}")

# ===========================================================================
# 5. GRÁFICO 2: DIAGRAMA DE CONSTELAÇÃO
# ===========================================================================
# Gera uma transmissão em SNR médio para visualização
ebno_constellation = 8.0  # dB
bits_vis = source([256, K])
cw_vis = encoder(bits_vis)
sym_vis = mapper(cw_vis)
no_vis = ebnodb2no(ebno_constellation, NUM_BITS_PER_SYMBOL, CODERATE)
noisy_vis = channel(sym_vis, no_vis)

# Converte tensores para numpy
sym_tx = sym_vis.detach().cpu().numpy().flatten() if hasattr(sym_vis, 'detach') else np.array(sym_vis).flatten()
sym_rx = noisy_vis.detach().cpu().numpy().flatten() if hasattr(noisy_vis, 'detach') else np.array(noisy_vis).flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Constelação transmitida
axes[0].scatter(sym_tx.real, sym_tx.imag, s=1, alpha=0.3, color="#2196F3")
axes[0].set_title(f"Constelacao Transmitida\n{2**NUM_BITS_PER_SYMBOL}-QAM", fontsize=13)
axes[0].set_xlabel("In-Phase (I)", fontsize=11)
axes[0].set_ylabel("Quadrature (Q)", fontsize=11)
axes[0].set_aspect("equal")
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color="k", linewidth=0.5)
axes[0].axvline(x=0, color="k", linewidth=0.5)

# Constelação recebida (com ruído)
axes[1].scatter(sym_rx.real, sym_rx.imag, s=1, alpha=0.3, color="#F44336")
axes[1].set_title(f"Constelacao Recebida (AWGN)\nEb/N0 = {ebno_constellation} dB", fontsize=13)
axes[1].set_xlabel("In-Phase (I)", fontsize=11)
axes[1].set_ylabel("Quadrature (Q)", fontsize=11)
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color="k", linewidth=0.5)
axes[1].axvline(x=0, color="k", linewidth=0.5)

plt.suptitle("Diagramas de Constelacao - Sistema LDPC + 16-QAM sobre AWGN", fontsize=14, y=1.02)
plt.tight_layout()
constellation_path = os.path.join(OUTPUT_DIR, "constellation.png")
plt.savefig(constellation_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Constelacoes salvas em: {constellation_path}")

# ===========================================================================
# 6. RESUMO DOS RESULTADOS
# ===========================================================================
print("\n[4/4] Resumo dos resultados:")
print("-" * 50)
print(f" {'Eb/N0 (dB)':>12} | {'BER':>12}")
print("-" * 50)
for snr, ber in zip(snr_simulated, ber_results):
    ber_str = f"{ber:.2e}" if ber > 0 else "0.00e+00"
    print(f" {snr:>+12.1f} | {ber_str:>12}")
print("-" * 50)

# Encontra o Eb/N0 onde BER <= 10^-4 (se disponível)
target_ber = 1e-4
idx_below = np.where(ber_results <= target_ber)[0]
if len(idx_below) > 0:  
    print(f"\n BER <= {target_ber:.0e} alcancado em Eb/N0 ~= {snr_simulated[idx_below[0]]:.1f} dB")
else:
    print(f"\n BER <= {target_ber:.0e} nao alcancado na faixa simulada")

print(f"\n Gráficos salvos em: {OUTPUT_DIR}")
print("=" * 65)
print(" Simulacao concluida com sucesso!")
print("=" * 65)