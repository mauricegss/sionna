import torch
import numpy as np
import os

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

model = Model(domain="freq",
              direction="uplink",
              cdl_model="C",
              delay_spread=100e-9,
              perfect_csi=False,
              speed=0.0,
              cyclic_prefix_length=6,
              pilot_ofdm_symbol_indices=[2, 11])

batch_size = 2
ebno_db = 15.0

no = ebnodb2no(ebno_db, model._num_bits_per_symbol, model._coderate, model._rg)
b = model._binary_source([batch_size, 1, model._num_streams_per_tx, model._k])
c = model._encoder(b)
x = model._mapper(c)
x_rg = model._rg_mapper(x)

cir = model._cdl(batch_size, model._rg.num_ofdm_symbols, 1/model._rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(model._frequencies, *cir, normalize=True)
y = model._channel_freq(x_rg, h_freq, no)

h_hat, err_var = model._ls_est(y, no)
x_hat, no_eff = model._lmmse_equ(y, h_hat, err_var, no)

print("y shape:", y.shape)
print("h_hat shape:", h_hat.shape)
print("x_hat shape:", x_hat.shape)
print("x shape:", x.shape)
