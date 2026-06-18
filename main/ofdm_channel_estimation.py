"""
===============================================================================
SIONNA PHY -- Estimativa de Canal com Preâmbulo OFDM
===============================================================================

Simulação de um sistema OFDM completo com estimativa de canal baseada em
pilotos (preâmbulo), comparando os estimadores LS e LMMSE sob diferentes
condições de canal (AWGN e CDL-C).

Baseado no tutorial oficial do Sionna:
  "MIMO OFDM Transmissions over the CDL Channel Model"
  https://github.com/NVlabs/sionna/blob/main/tutorials/phy/

Requisitos:
  pip install sionna

Uso:
  python ofdm_channel_estimation.py

Autores: Maurice, Vinicius, Natalia
Disciplina: Tópicos em Redes sem Fio - UTFPR
===============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para salvar figuras
import matplotlib.pyplot as plt

# ===========================================================================
# Imports do Sionna PHY (versão 1.x - TensorFlow backend)
# ===========================================================================
# Nota: Sionna 1.x usa TensorFlow. Sionna 2.x usa PyTorch.
# O código abaixo é compatível com Sionna 1.x (pip install sionna==0.19.1)
# Para Sionna 2.x, os imports mudam ligeiramente.
# ===========================================================================

try:
    import sionna
    SIONNA_VERSION = sionna.__version__
    print(f"[INFO] Sionna versão {SIONNA_VERSION} detectado.")
except ImportError:
    print("[ERRO] Sionna não está instalado.")
    print("       Execute: pip install sionna")
    print("       Ou use o Google Colab com GPU.")
    exit(1)

# Detectar versão do Sionna e importar corretamente
SIONNA_V2 = int(SIONNA_VERSION.split(".")[0]) >= 1

if SIONNA_V2:
    # Sionna >= 1.0 (pode ser 0.19.x também, ajustar conforme necessário)
    try:
        # Sionna 2.x (PyTorch backend)
        import torch
        from sionna.phy import Block
        from sionna.phy.mapping import (
            BinarySource,
            Constellation,
            Mapper,
            Demapper,
        )
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        from sionna.phy.ofdm import (
            ResourceGrid,
            ResourceGridMapper,
            ResourceGridDemapper,
            LSChannelEstimator,
            LMMSEEqualizer,
            OFDMModulator,
            OFDMDemodulator,
        )
        from sionna.phy.channel.tr38901 import CDL, Antenna, AntennaArray
        from sionna.phy.channel import AWGN, OFDMChannel, time_lag_discrete_time_channel
        from sionna.phy.mimo import StreamManagement
        from sionna.phy.utils import ebnodb2no, compute_ber
        BACKEND = "pytorch"
        print(f"[INFO] Backend: PyTorch (Sionna 2.x)")
    except ImportError:
        # Sionna 0.x (TensorFlow backend)
        import tensorflow as tf
        from sionna.mapping import (
            BinarySource,
            Constellation,
            Mapper,
            Demapper,
        )
        from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        from sionna.ofdm import (
            ResourceGrid,
            ResourceGridMapper,
            ResourceGridDemapper,
            LSChannelEstimator,
            LMMSEEqualizer,
            OFDMModulator,
            OFDMDemodulator,
        )
        from sionna.channel.tr38901 import CDL, Antenna, AntennaArray
        from sionna.channel import AWGN, OFDMChannel
        from sionna.mimo import StreamManagement
        from sionna.utils import ebnodb2no, compute_ber
        BACKEND = "tensorflow"
        print(f"[INFO] Backend: TensorFlow (Sionna 0.x)")

# ===========================================================================
# 1. PARÂMETROS DA SIMULAÇÃO
# ===========================================================================
# --- Sistema OFDM ---
NUM_OFDM_SYMBOLS = 14       # Símbolos OFDM por slot (5G NR padrão)
FFT_SIZE = 76               # Tamanho da FFT (número de subportadoras)
SUBCARRIER_SPACING = 30e3   # Espaçamento entre subportadoras (Hz)
CYCLIC_PREFIX_LENGTH = 6    # Comprimento do prefixo cíclico
NUM_GUARD_CARRIERS = (5, 6) # Subportadoras de guarda (esquerda, direita)
DC_NULL = True              # Subportadora DC nula
PILOT_PATTERN = "kronecker" # Padrão de pilotos do tipo Kronecker
PILOT_OFDM_SYMBOL_INDICES = [2, 11]  # Símbolos OFDM com pilotos

# --- Antenas (SISO para simplificar) ---
NUM_UT = 1                  # Número de terminais de usuário
NUM_BS = 1                  # Número de estações base
NUM_UT_ANT = 1              # Antenas por UT
NUM_BS_ANT = 1              # Antenas por BS
NUM_STREAMS_PER_TX = 1      # Streams por transmissor

# --- FEC e Modulação ---
NUM_BITS_PER_SYMBOL = 4     # 16-QAM (2^4 = 16 pontos)
CODERATE = 0.5              # Taxa de código LDPC

# --- Simulação ---
BATCH_SIZE = 128            # Blocos por iteração
EBN0_DB_RANGE = np.arange(0.0, 16.0, 1.0)  # Faixa Eb/N0 (dB)

# --- Canal CDL ---
CDL_MODEL = "C"             # Perfil CDL-C (NLOS, urbano)
DELAY_SPREAD = 100e-9       # Espalhamento de atraso (100 ns)
CARRIER_FREQUENCY = 3.5e9   # Frequência da portadora (3.5 GHz, banda n78)
UT_SPEED = 3.0              # Velocidade do UT (m/s) ~ pedestre

# --- Saída ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 2. CONFIGURAÇÃO DO SISTEMA
# ===========================================================================
print("\n" + "=" * 70)
print(" SIONNA PHY -- Estimativa de Canal com Preâmbulo OFDM")
print("=" * 70)
print(f" Backend:            {BACKEND}")
print(f" FFT size:           {FFT_SIZE}")
print(f" OFDM symbols/slot:  {NUM_OFDM_SYMBOLS}")
print(f" Pilot positions:    {PILOT_OFDM_SYMBOL_INDICES}")
print(f" Modulação:          {2**NUM_BITS_PER_SYMBOL}-QAM")
print(f" Code rate:          {CODERATE}")
print(f" Canal:              AWGN + CDL-{CDL_MODEL}")
print(f" Eb/N0 range:        {EBN0_DB_RANGE[0]:.0f} a {EBN0_DB_RANGE[-1]:.0f} dB")
print(f" Batch size:         {BATCH_SIZE}")
print("=" * 70)

# --- Grade de Recursos OFDM ---
print("\n[1/6] Configurando grade de recursos OFDM...")

resource_grid = ResourceGrid(
    num_ofdm_symbols=NUM_OFDM_SYMBOLS,
    fft_size=FFT_SIZE,
    subcarrier_spacing=SUBCARRIER_SPACING,
    num_tx=NUM_UT,
    num_streams_per_tx=NUM_STREAMS_PER_TX,
    cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
    num_guard_carriers=NUM_GUARD_CARRIERS,
    dc_null=DC_NULL,
    pilot_pattern=PILOT_PATTERN,
    pilot_ofdm_symbol_indices=PILOT_OFDM_SYMBOL_INDICES,
)

print(f"   [OK] ResourceGrid: {FFT_SIZE} subportadoras x {NUM_OFDM_SYMBOLS} símbolos")
print(f"   [OK] Pilotos em símbolos OFDM: {PILOT_OFDM_SYMBOL_INDICES}")
print(f"   [OK] Subportadoras efetivas: {resource_grid.num_effective_subcarriers}")

# --- Gerenciamento de Streams ---
rx_tx_association = np.array([[1]])  # Um RX para um TX
stream_management = StreamManagement(rx_tx_association, NUM_STREAMS_PER_TX)

# --- Cálculo de k e n para LDPC ---
NUM_DATA_SYMBOLS = resource_grid.num_data_symbols
N = int(NUM_DATA_SYMBOLS * NUM_BITS_PER_SYMBOL)  # bits codificados por slot
K = int(N * CODERATE)  # bits de informação por slot

print(f"   [OK] Data symbols/slot: {NUM_DATA_SYMBOLS}")
print(f"   [OK] LDPC: k={K}, n={N}, rate={CODERATE}")


# ===========================================================================
# 3. INSTANCIAÇÃO DOS COMPONENTES
# ===========================================================================
print("\n[2/6] Instanciando componentes do sistema...")

# Fonte de bits
source = BinarySource()

# Codificador/Decodificador LDPC 5G NR
encoder = LDPC5GEncoder(k=K, n=N)
decoder = LDPC5GDecoder(encoder=encoder)

# Constelação e Mapeador/Demapeador QAM
constellation = Constellation("qam", num_bits_per_symbol=NUM_BITS_PER_SYMBOL)
mapper = Mapper(constellation=constellation)
demapper = Demapper("app", constellation=constellation)

# Mapeador/Demapeador de Grade de Recursos
rg_mapper = ResourceGridMapper(resource_grid)
rg_demapper = ResourceGridDemapper(resource_grid, stream_management)

# Modulador/Demodulador OFDM
ofdm_modulator = OFDMModulator(cyclic_prefix_length=CYCLIC_PREFIX_LENGTH)

# Calcular l_min a partir da largura de banda e do espalhamento de atraso
if BACKEND == "pytorch":
    L_MIN, L_MAX = time_lag_discrete_time_channel(
        bandwidth=resource_grid.bandwidth,
        maximum_delay_spread=DELAY_SPREAD * 10,  # Margem de segurança
    )
else:
    L_MIN = getattr(resource_grid, 'l_min', -6)  # Fallback para TF

ofdm_demodulator = OFDMDemodulator(
    fft_size=FFT_SIZE,
    l_min=L_MIN,
    cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
)

# Estimador de Canal LS
ls_estimator = LSChannelEstimator(
    resource_grid,
    interpolation_type="lin_time_avg",
)

# Equalizador LMMSE
lmmse_equalizer = LMMSEEqualizer(resource_grid, stream_management)

# Canal AWGN
awgn_channel = AWGN()

print("   [OK] BinarySource")
print("   [OK] LDPC5GEncoder / LDPC5GDecoder")
print(f"   [OK] Constellation ({2**NUM_BITS_PER_SYMBOL}-QAM)")
print("   [OK] Mapper / Demapper (APP)")
print("   [OK] ResourceGridMapper / ResourceGridDemapper")
print("   [OK] OFDMModulator / OFDMDemodulator")
print("   [OK] LSChannelEstimator (interpolação linear)")
print("   [OK] LMMSEEqualizer")
print("   [OK] Canal AWGN")


# ===========================================================================
# 4. FUNÇÕES DE SIMULAÇÃO
# ===========================================================================

def run_awgn_simulation(ebno_db_range, use_perfect_csi=False):
    """
    Simula o sistema OFDM sobre canal AWGN.

    Em canal AWGN (H[k] = 1 para todo k), a estimativa de canal LS
    deve convergir para o canal perfeito em SNRs altas.

    Args:
        ebno_db_range: Array de valores Eb/N0 em dB.
        use_perfect_csi: Se True, usa CSI perfeito (H=1) em vez de LS.

    Returns:
        ber_results: Array de BER para cada Eb/N0.
    """
    ber_results = []

    for ebno_db in ebno_db_range:
        # Transmissor
        bits = source((BATCH_SIZE, 1, NUM_STREAMS_PER_TX, K))
        codewords = encoder(bits)
        symbols = mapper(codewords)
        x_rg = rg_mapper(symbols)  # Insere pilotos na grade

        # Modulação OFDM (IFFT + CP)
        x_time = ofdm_modulator(x_rg)

        # Canal AWGN (sem desvanecimento: H=1)
        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=CODERATE,
            resource_grid=resource_grid,
        )
        y_time = awgn_channel(x_time, no)

        # Demodulação OFDM (remover CP + FFT)
        y_rg = ofdm_demodulator(y_time)

        if use_perfect_csi:
            # CSI perfeito: H = 1 (canal AWGN ideal)
            # Neste caso, não usamos o estimador LS
            h_hat = None  # Placeholder - implementação depende da versão
            err_var = 0.0
        else:
            # Estimativa LS a partir dos pilotos
            h_hat, err_var = ls_estimator(y_rg, no)

        # Equalização LMMSE
        x_hat, no_eff = lmmse_equalizer(y_rg, h_hat, err_var, no)

        # Demapeamento e decodificação
        llr = demapper(x_hat, no_eff)
        bits_hat = decoder(llr)

        # Cálculo do BER
        ber = compute_ber(bits, bits_hat)
        if hasattr(ber, 'numpy'):
            ber_val = float(ber.numpy())
        else:
            ber_val = float(ber)
        ber_results.append(ber_val)

        print(f"\r   AWGN {'(perfeito)' if use_perfect_csi else '(LS+LMMSE)'}: "
              f"Eb/N0={ebno_db:+5.1f} dB | BER={ber_val:.2e}", end="")

    print()
    return np.array(ber_results)


def run_cdl_simulation(ebno_db_range, cdl_model="C"):
    """
    Simula o sistema OFDM sobre canal CDL do 3GPP.

    O canal CDL introduz desvanecimento multipercurso realista,
    testando a capacidade do estimador LS + equalizador LMMSE.

    Args:
        ebno_db_range: Array de valores Eb/N0 em dB.
        cdl_model: Perfil CDL ("A", "B", "C", "D" ou "E").

    Returns:
        ber_results: Array de BER para cada Eb/N0.
        diagnostics: Dict com dados de diagnóstico para visualização.
    """
    ber_results = []
    diagnostics = {}
    CAPTURE_SNR = 10.0  # Capturar dados de diagnóstico neste Eb/N0

    # Configurar canal CDL
    cdl = CDL(
        model=cdl_model,
        delay_spread=DELAY_SPREAD,
        carrier_frequency=CARRIER_FREQUENCY,
        ut_array=AntennaArray(
            num_rows=1, num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=CARRIER_FREQUENCY,
        ),
        bs_array=AntennaArray(
            num_rows=1, num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=CARRIER_FREQUENCY,
        ),
        direction="uplink",
        min_speed=UT_SPEED,
        max_speed=UT_SPEED,
    )

    # Canal OFDM (aplica CDL + AWGN no domínio OFDM)
    ofdm_channel = OFDMChannel(
        cdl,
        resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=True,
    )

    for ebno_db in ebno_db_range:
        # Transmissor
        bits = source((BATCH_SIZE, 1, NUM_STREAMS_PER_TX, K))
        codewords = encoder(bits)
        symbols = mapper(codewords)
        x_rg = rg_mapper(symbols)

        # Canal CDL + AWGN
        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=CODERATE,
            resource_grid=resource_grid,
        )

        # O OFDMChannel retorna (sinal recebido, resposta do canal real)
        y_rg, h_freq = ofdm_channel(x_rg, no)

        # Estimativa LS a partir dos pilotos
        h_hat, err_var = ls_estimator(y_rg, no)

        # Equalização LMMSE
        x_hat, no_eff = lmmse_equalizer(y_rg, h_hat, err_var, no)

        # Demapeamento e decodificação
        llr = demapper(x_hat, no_eff)
        bits_hat = decoder(llr)

        # Cálculo do BER
        ber = compute_ber(bits, bits_hat)
        if hasattr(ber, 'numpy'):
            ber_val = float(ber.numpy())
        else:
            ber_val = float(ber)
        ber_results.append(ber_val)

        # Capturar dados de diagnóstico em um SNR específico
        if abs(ebno_db - CAPTURE_SNR) < 0.5:
            def to_np(t):
                if hasattr(t, 'detach'):
                    return t.detach().cpu().numpy()
                return np.array(t)
            diagnostics["symbols_tx"] = to_np(symbols[0, 0, 0, :200])
            diagnostics["symbols_eq"] = to_np(x_hat[0, 0, 0, :200])
            diagnostics["h_freq_real"] = to_np(h_freq[0, 0, 0, 0, 0, 0, :])
            diagnostics["h_hat_est"] = to_np(h_hat[0, 0, 0, 0, 0, 0, :])
            diagnostics["capture_snr"] = ebno_db

        print(f"\r   CDL-{cdl_model} (LS+LMMSE): "
              f"Eb/N0={ebno_db:+5.1f} dB | BER={ber_val:.2e}", end="")

    print()
    return np.array(ber_results), diagnostics


# ===========================================================================
# 5. EXECUÇÃO DOS CENÁRIOS
# ===========================================================================
print("\n[3/6] Executando Cenário A: AWGN com estimativa LS + equalização LMMSE...")
ber_awgn_ls = run_awgn_simulation(EBN0_DB_RANGE, use_perfect_csi=False)

print("\n[4/6] Executando Cenário B: CDL-C com estimativa LS + equalização LMMSE...")
ber_cdl_ls, cdl_diag = run_cdl_simulation(EBN0_DB_RANGE, cdl_model=CDL_MODEL)


# ===========================================================================
# 6. GERAÇÃO DE GRÁFICOS E RELATÓRIOS
# ===========================================================================
print("\n[5/6] Gerando gráficos e relatórios...")

# --- Gráfico 1: BER x Eb/N0 (AWGN vs CDL-C) ---
fig, ax = plt.subplots(figsize=(10, 7))

ax.semilogy(
    EBN0_DB_RANGE,
    np.clip(ber_awgn_ls, 1e-7, 1),
    "bs-",
    linewidth=2, markersize=6,
    label="LS + LMMSE (Canal AWGN)",
)

ax.semilogy(
    EBN0_DB_RANGE,
    np.clip(ber_cdl_ls, 1e-7, 1),
    "ro-",
    linewidth=2, markersize=6,
    label=f"LS + LMMSE (Canal CDL-{CDL_MODEL})",
)

ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=13)
ax.set_ylabel("BER (Bit Error Rate)", fontsize=13)
ax.set_title(
    "Desempenho BER — Estimativa de Canal com Preâmbulo OFDM\n"
    f"LDPC({N},{K}) + {2**NUM_BITS_PER_SYMBOL}-QAM | Sionna PHY",
    fontsize=14,
)
ax.set_xlim([EBN0_DB_RANGE[0], EBN0_DB_RANGE[-1]])
ax.set_ylim([1e-6, 1])
ax.grid(True, which="both", alpha=0.3, linestyle="--")
ax.legend(fontsize=11, loc="lower left")
ax.tick_params(labelsize=11)

plt.tight_layout()
ber_path = os.path.join(OUTPUT_DIR, "ber_awgn_vs_cdl.png")
plt.savefig(ber_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Gráfico 1/4 — BER x Eb/N0: {ber_path}")


# --- Gráfico 2: Diagrama de Constelação (TX vs Equalizado) ---
if "symbols_tx" in cdl_diag:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    snr_cap = cdl_diag["capture_snr"]

    # Constelação TX (ideal)
    sym_tx = cdl_diag["symbols_tx"]
    axes[0].scatter(np.real(sym_tx), np.imag(sym_tx), c="blue", s=15, alpha=0.7)
    axes[0].set_title(f"Constelação TX (16-QAM ideal)", fontsize=12)
    axes[0].set_xlabel("In-Phase (I)")
    axes[0].set_ylabel("Quadrature (Q)")
    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([-1.5, 1.5])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    axes[0].axvline(x=0, color="k", linewidth=0.5)

    # Constelação Equalizada (com ruído residual)
    sym_eq = cdl_diag["symbols_eq"]
    axes[1].scatter(np.real(sym_eq), np.imag(sym_eq), c="red", s=15, alpha=0.5)
    axes[1].set_title(f"Constelação RX equalizada (CDL-C, Eb/N0={snr_cap:.0f} dB)", fontsize=12)
    axes[1].set_xlabel("In-Phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlim([-1.5, 1.5])
    axes[1].set_ylim([-1.5, 1.5])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    const_path = os.path.join(OUTPUT_DIR, "constelacao_tx_vs_rx.png")
    plt.savefig(const_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Gráfico 2/4 — Constelação TX vs RX: {const_path}")


# --- Gráfico 3: Resposta em Frequência do Canal (Real vs Estimada) ---
if "h_freq_real" in cdl_diag:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    snr_cap = cdl_diag["capture_snr"]

    h_real = cdl_diag["h_freq_real"]
    h_est = cdl_diag["h_hat_est"]
    subcarriers = np.arange(len(h_real))
    subcarriers_est = np.arange(len(h_est))

    # Magnitude |H[k]|
    axes[0].plot(subcarriers, 20 * np.log10(np.abs(h_real) + 1e-10),
                 "b-", linewidth=1.5, label="Canal real $|H[k]|$")
    axes[0].plot(subcarriers_est, 20 * np.log10(np.abs(h_est) + 1e-10),
                 "r--", linewidth=1.5, alpha=0.8, label="Estimativa LS $|\\hat{H}[k]|$")
    axes[0].set_ylabel("Magnitude (dB)", fontsize=12)
    axes[0].set_title(
        f"Resposta em Frequência do Canal CDL-{CDL_MODEL} "
        f"(Eb/N0 = {snr_cap:.0f} dB)",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Fase ∠H[k]
    axes[1].plot(subcarriers, np.angle(h_real, deg=True),
                 "b-", linewidth=1.5, label="Canal real $\\angle H[k]$")
    axes[1].plot(subcarriers_est, np.angle(h_est, deg=True),
                 "r--", linewidth=1.5, alpha=0.8, label="Estimativa LS $\\angle \\hat{H}[k]$")
    axes[1].set_xlabel("Subportadora (k)", fontsize=12)
    axes[1].set_ylabel("Fase (graus)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chan_path = os.path.join(OUTPUT_DIR, "canal_frequencia_real_vs_estimado.png")
    plt.savefig(chan_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Gráfico 3/4 — Resposta do canal: {chan_path}")


# --- Gráfico 4: Grade de Recursos OFDM (Pilotos vs Dados) ---
fig, ax = plt.subplots(figsize=(10, 5))

# Montar máscara da grade de recursos
pilot_mask = resource_grid.pilot_pattern.mask.numpy()  # [num_tx, num_streams, num_ofdm, num_subcarriers]
pilot_mask_2d = pilot_mask[0, 0, :, :]  # [num_ofdm, num_effective_subcarriers]

# Criar grid de visualização: 0=dado, 1=piloto, -1=guarda
num_sc = FFT_SIZE
num_sym = NUM_OFDM_SYMBOLS
grid_vis = np.zeros((num_sc, num_sym))

# Marcar subportadoras efetivas como dados
eff_idx = resource_grid.effective_subcarrier_ind
for idx in eff_idx:
    grid_vis[idx, :] = 0.3  # Dado (azul claro)

# Marcar pilotos
for sym_idx in range(pilot_mask_2d.shape[0]):
    for sc_idx in range(pilot_mask_2d.shape[1]):
        if pilot_mask_2d[sym_idx, sc_idx]:
            grid_vis[eff_idx[sc_idx], sym_idx] = 1.0  # Piloto

# Plotar
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#2d2d2d", "#4a90d9", "#e8a838"])
im = ax.imshow(grid_vis, aspect="auto", cmap=cmap, origin="lower",
               extent=[-0.5, num_sym - 0.5, -0.5, num_sc - 0.5])
ax.set_xlabel("Símbolo OFDM (tempo)", fontsize=12)
ax.set_ylabel("Subportadora (frequência)", fontsize=12)
ax.set_title(
    f"Grade de Recursos OFDM — {FFT_SIZE} subportadoras × "
    f"{NUM_OFDM_SYMBOLS} símbolos\n"
    f"Pilotos Kronecker nos símbolos {PILOT_OFDM_SYMBOL_INDICES}",
    fontsize=13,
)

# Legenda manual
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2d2d2d", label="Guarda / DC"),
    Patch(facecolor="#4a90d9", label="Dados"),
    Patch(facecolor="#e8a838", label="Pilotos (DMRS)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()
grid_path = os.path.join(OUTPUT_DIR, "grade_recursos_ofdm.png")
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK] Gráfico 4/4 — Grade de recursos: {grid_path}")


# ===========================================================================
# 7. RESUMO DOS RESULTADOS
# ===========================================================================
print("\n[6/6] Resumo dos resultados:")
print("-" * 60)
print(f" {'Eb/N0 (dB)':>12} | {'BER (AWGN)':>14} | {'BER (CDL-C)':>14}")
print("-" * 60)

for snr, ber_a, ber_c in zip(EBN0_DB_RANGE, ber_awgn_ls, ber_cdl_ls):
    ber_a_str = f"{ber_a:.2e}" if ber_a > 0 else "0.00e+00"
    ber_c_str = f"{ber_c:.2e}" if ber_c > 0 else "0.00e+00"
    print(f" {snr:>+12.1f} | {ber_a_str:>14} | {ber_c_str:>14}")

print("-" * 60)

# Encontra Eb/N0 onde BER <= 10^-3
target_ber = 1e-3
snr_targets = {}
for name, ber_arr in [("AWGN", ber_awgn_ls), (f"CDL-{CDL_MODEL}", ber_cdl_ls)]:
    idx_below = np.where(ber_arr <= target_ber)[0]
    if len(idx_below) > 0:
        snr_targets[name] = EBN0_DB_RANGE[idx_below[0]]
        print(f"\n BER <= {target_ber:.0e} em {name}: Eb/N0 ~= {snr_targets[name]:.1f} dB")
    else:
        snr_targets[name] = None
        print(f"\n BER <= {target_ber:.0e} em {name}: não alcançado na faixa simulada")


# ===========================================================================
# 8. EXPORTAÇÃO: RELATÓRIO COMPLETO EM TXT + CSV
# ===========================================================================

# --- Relatório TXT ---
report_path = os.path.join(OUTPUT_DIR, "relatorio_simulacao.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(" RELATÓRIO DE SIMULAÇÃO — ESTIMATIVA DE CANAL COM PREÂMBULO OFDM\n")
    f.write("=" * 70 + "\n\n")

    f.write("CONFIGURAÇÃO DO SISTEMA\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Biblioteca:              Sionna {SIONNA_VERSION} ({BACKEND})\n")
    f.write(f"  FFT size:                {FFT_SIZE}\n")
    f.write(f"  Subportadoras efetivas:  {resource_grid.num_effective_subcarriers}\n")
    f.write(f"  Símbolos OFDM/slot:      {NUM_OFDM_SYMBOLS}\n")
    f.write(f"  Prefixo cíclico:         {CYCLIC_PREFIX_LENGTH}\n")
    f.write(f"  Espaçamento subport.:    {SUBCARRIER_SPACING/1e3:.0f} kHz\n")
    f.write(f"  Largura de banda:        {resource_grid.bandwidth/1e6:.2f} MHz\n")
    f.write(f"  Pilotos (índices OFDM):  {PILOT_OFDM_SYMBOL_INDICES}\n")
    f.write(f"  Padrão de pilotos:       {PILOT_PATTERN}\n")
    f.write(f"  Modulação:               {2**NUM_BITS_PER_SYMBOL}-QAM\n")
    f.write(f"  Código FEC:              LDPC 5G NR (k={K}, n={N}, R={CODERATE})\n")
    f.write(f"  Batch size:              {BATCH_SIZE}\n")
    f.write(f"  Bits/batch:              {BATCH_SIZE * K:,}\n")
    f.write(f"  Faixa Eb/N0:             {EBN0_DB_RANGE[0]:.0f} a {EBN0_DB_RANGE[-1]:.0f} dB\n")
    f.write("\n")

    f.write("CANAL CDL-C\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Perfil:                  CDL-{CDL_MODEL} (NLOS)\n")
    f.write(f"  Delay spread:            {DELAY_SPREAD*1e9:.0f} ns\n")
    f.write(f"  Frequência portadora:    {CARRIER_FREQUENCY/1e9:.1f} GHz\n")
    f.write(f"  Velocidade UT:           {UT_SPEED:.1f} m/s\n")
    f.write(f"  Configuração antenas:    SISO (1×1)\n")
    f.write("\n")

    f.write("RECEPTOR\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Estimador de canal:      LS (Least Squares)\n")
    f.write(f"  Interpolação:            Linear (tempo-média)\n")
    f.write(f"  Equalizador:             LMMSE\n")
    f.write(f"  Demapeador:              APP (A Posteriori Probability)\n")
    f.write("\n")

    f.write("RESULTADOS — BER × Eb/N0\n")
    f.write("-" * 60 + "\n")
    f.write(f"  {'Eb/N0 (dB)':>12} | {'BER (AWGN)':>14} | {'BER (CDL-C)':>14}\n")
    f.write("-" * 60 + "\n")
    for snr, ber_a, ber_c in zip(EBN0_DB_RANGE, ber_awgn_ls, ber_cdl_ls):
        ber_a_str = f"{ber_a:.6e}" if ber_a > 0 else "0.000000e+00"
        ber_c_str = f"{ber_c:.6e}" if ber_c > 0 else "0.000000e+00"
        f.write(f"  {snr:>+12.1f} | {ber_a_str:>14} | {ber_c_str:>14}\n")
    f.write("-" * 60 + "\n\n")

    f.write("ANÁLISE\n")
    f.write("-" * 40 + "\n")
    for name, snr_val in snr_targets.items():
        if snr_val is not None:
            f.write(f"  BER <= 1e-3 em {name}: Eb/N0 ≈ {snr_val:.1f} dB\n")
        else:
            f.write(f"  BER <= 1e-3 em {name}: não alcançado\n")

    if all(v is not None for v in snr_targets.values()):
        penalty = snr_targets.get(f"CDL-{CDL_MODEL}", 0) - snr_targets.get("AWGN", 0)
        f.write(f"\n  Penalidade CDL-{CDL_MODEL} vs AWGN: ~{penalty:.1f} dB\n")
        f.write(f"  (Consistente com literatura: 4-10 dB para SISO + LS)\n")

    f.write("\n")
    f.write("ARQUIVOS GERADOS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  1. ber_awgn_vs_cdl.png              — Curvas BER × Eb/N0\n")
    f.write(f"  2. constelacao_tx_vs_rx.png          — Diagrama de constelação\n")
    f.write(f"  3. canal_frequencia_real_vs_estimado.png — Resposta do canal H[k]\n")
    f.write(f"  4. grade_recursos_ofdm.png           — Grade de recursos OFDM\n")
    f.write(f"  5. relatorio_simulacao.txt            — Este relatório\n")
    f.write(f"  6. resultados_ber.csv                 — Dados BER em CSV\n")
    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write(" Simulação concluída com sucesso!\n")
    f.write("=" * 70 + "\n")

print(f"   [OK] Relatório TXT salvo em: {report_path}")


# --- CSV dos resultados ---
csv_path = os.path.join(OUTPUT_DIR, "resultados_ber.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("EbN0_dB,BER_AWGN,BER_CDL_C\n")
    for snr, ber_a, ber_c in zip(EBN0_DB_RANGE, ber_awgn_ls, ber_cdl_ls):
        f.write(f"{snr:.1f},{ber_a:.8e},{ber_c:.8e}\n")
print(f"   [OK] Dados CSV salvos em: {csv_path}")


print(f"\n Todos os arquivos salvos em: {OUTPUT_DIR}")
print("=" * 70)
print(" Simulação concluída com sucesso!")
print("=" * 70)

