# ========================================================================
# CÓDIGO OFICIAL E INALTERADO DA NVIDIA
# Extraído de MIMO_OFDM_Transmissions_over_CDL.ipynb
# ========================================================================

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    import os
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Set random seed for reproducibility
sionna.phy.config.seed = 42

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                               time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                               OFDMChannel, TimeChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber

# Define the number of UT and BS antennas.
# For the CDL model, that will be used in this notebook, only
# a single UT and BS are supported.
num_ut = 1
num_bs = 1
num_ut_ant = 4
num_bs_ant = 8

# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = num_ut_ant

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=76,
                  subcarrier_spacing=15e3,
                  num_tx=1,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=6,
                  num_guard_carriers=[5,6],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
rg.show();

rg.pilot_pattern.show();

plt.figure()
plt.title("Real Part of the Pilot Sequences")
for i in range(num_streams_per_tx):
    plt.stem(np.real(rg.pilot_pattern.pilots[0, i].cpu().numpy()),
             markerfmt="C{}.".format(i), linefmt="C{}-".format(i),
             label="Stream {}".format(i))
plt.legend()
print("Average energy per pilot symbol: {:1.2f}".format(np.mean(np.abs(rg.pilot_pattern.pilots[0,0].cpu().numpy())**2)))

carrier_frequency = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

ut_array = AntennaArray(num_rows=1,
                        num_cols=int(num_ut_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
ut_array.show()

bs_array = AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
bs_array.show()

ut_array.show_element_radiation_pattern()

delay_spread = 300e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                      # about how to choose this value.

direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                      # In the `uplink`, the UT is transmitting.
cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]

speed = 10            # UT speed [m/s]. BSs are always assumed to be fixed.
                      # The direction of travel will chosen randomly within the x-y plane.

# Configure a channel impulse reponse (CIR) generator for the CDL model.
# cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)

a, tau = cdl(batch_size=32, num_time_steps=rg.num_ofdm_symbols, sampling_frequency=1/rg.ofdm_symbol_duration)

print("Shape of the path gains: ", a.shape)
print("Shape of the delays:", tau.shape)

a_ = a.cpu().numpy()
tau_ = tau.cpu().numpy()

plt.figure()
plt.title("Channel impulse response realization")
plt.stem(tau_[0,0,0,:]/1e-9, np.abs(a_)[0,0,0,0,0,:,0])
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$")


plt.figure()
plt.title("Time evolution of path gain")
plt.plot(np.arange(rg.num_ofdm_symbols)*rg.ofdm_symbol_duration/1e-6, np.real(a_)[0,0,0,0,0,0,:])
plt.plot(np.arange(rg.num_ofdm_symbols)*rg.ofdm_symbol_duration/1e-6, np.imag(a_)[0,0,0,0,0,0,:])
plt.legend(["Real part", "Imaginary part"])

plt.xlabel(r"$t$ [us]")
plt.ylabel(r"$a$");

frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

plt.figure()
plt.title("Channel frequency response")
plt.plot(np.real(h_freq[0,0,0,0,0,0,:].cpu().numpy()))
plt.plot(np.imag(h_freq[0,0,0,0,0,0,:].cpu().numpy()))
plt.xlabel("OFDM Symbol Index")
plt.ylabel(r"$h$")
plt.legend(["Real part", "Imaginary part"]);

# Function that will apply the channel frequency response to an input signal
channel_freq = ApplyOFDMChannel(add_awgn=True)

# The following values for truncation are recommended.
# Please feel free to tailor them to you needs.
l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
l_tot = l_max-l_min+1

a, tau = cdl(batch_size=2, num_time_steps=rg.num_time_samples+l_tot-1, sampling_frequency=rg.bandwidth)

h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)

plt.figure()
plt.title("Discrete-time channel impulse response")
plt.stem(np.abs(h_time[0,0,0,0,0,0].cpu().numpy()))
plt.xlabel(r"Time step $\ell$")
plt.ylabel(r"$|\bar{h}|$");

# Function that will apply the discrete-time channel impulse response to an input signal
channel_time = ApplyTimeChannel(rg.num_time_samples, l_tot=l_tot, add_awgn=True)

num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # Code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits

# The binary source will create batches of information bits
binary_source = BinarySource()

# The encoder maps information bits to coded bits
encoder = LDPC5GEncoder(k, n)

# The mapper maps blocks of information bits to constellation symbols
mapper = Mapper("qam", num_bits_per_symbol)

# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = ResourceGridMapper(rg)

# The zero forcing precoder precodes the transmit stream towards the intended antennas
zf_precoder = RZFPrecoder(rg, sm, return_effective_channel=True)

# OFDM modulator and demodulator
modulator = OFDMModulator(rg.cyclic_prefix_length)
demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)

# This function removes nulled subcarriers from any tensor having the shape of a resource grid
remove_nulled_scs = RemoveNulledSubcarriers(rg)

# The LS channel estimator will provide channel estimates and error variances
ls_est = LSChannelEstimator(rg, interpolation_type="nn")

# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)

# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)

# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)

batch_size = 32 # Depending on the memory of your GPU (or system when a CPU is used),
                # you can in(de)crease the batch size. The larger the batch size, the
                # more memory is required. However, simulations will also run much faster.
ebno_db = 40
perfect_csi = False # Change to switch between perfect and imperfect CSI

# Compute the noise power for a given Eb/No value.
# This takes not only the coderate but also the overheads related pilot
# transmissions and nulled carriers
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)

# As explained above, we generate random batches of CIR, transform them
# in the frequency domain and apply them to the resource grid in the
# frequency domain.
cir = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
y = channel_freq(x_rg, h_freq, no)

if perfect_csi:
    # For perfect CSI, the receiver gets the channel frequency response as input
    # However, the channel estimator only computes estimates on the non-nulled
    # subcarriers. Therefore, we need to remove them here from `h_freq`.
    # This step can be skipped if no subcarriers are nulled.
    h_hat, err_var = remove_nulled_scs(h_freq), 0.
else:
    h_hat, err_var = ls_est (y, no)

x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))

ofdm_channel = OFDMChannel(cdl, rg, add_awgn=True, normalize_channel=True, return_channel=True)
y, h_freq = ofdm_channel(x_rg, no)

batch_size = 4 # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
ebno_db = 30
perfect_csi = True

no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)

# The CIR needs to be sampled every 1/bandwith [s].
# In contrast to frequency-domain modeling, this implies
# that the channel can change over the duration of a single
# OFDM symbol. We now also need to simulate more
# time steps.
cir = cdl(batch_size, rg.num_time_samples+l_tot-1, rg.bandwidth)

# OFDM modulation with cyclic prefix insertion
x_time = modulator(x_rg)

# Compute the discrete-time channel impulse reponse
h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)

# Compute the channel output
# This computes the full convolution between the time-varying
# discrete-time channel impulse reponse and the discrete-time
# transmit signal. With this technique, the effects of an
# insufficiently long cyclic prefix will become visible. This
# is in contrast to frequency-domain modeling which imposes
# no inter-symbol interfernce.
y_time = channel_time(x_time, h_time, no)

# OFDM demodulation and cyclic prefix removal
y = demodulator(y_time)

if perfect_csi:

    a, tau = cir

    # We need to sub-sample the channel impulse reponse to compute perfect CSI
    # for the receiver as it only needs one channel realization per OFDM symbol
    a_freq = a[...,rg.cyclic_prefix_length:-1:(rg.fft_size+rg.cyclic_prefix_length)]
    a_freq = a_freq[...,:rg.num_ofdm_symbols]

    # Compute the channel frequency response
    h_freq = cir_to_ofdm_channel(frequencies, a_freq, tau, normalize=True)

    h_hat, err_var = remove_nulled_scs(h_freq), 0.
else:
    h_hat, err_var = ls_est (y, no)

x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))

time_channel = TimeChannel(cdl, rg.bandwidth, rg.num_time_samples,
                           l_min=l_min, l_max=l_max, normalize_channel=True,
                           add_awgn=True, return_channel=True)

y_time, h_time = time_channel(x_time, no)

# In the example above, we assumed perfect CSI, i.e.,
# h_hat correpsond to the exact ideal channel frequency response.
h_perf = h_hat[0,0,0,0,0,0]

# We now compute the LS channel estimate from the pilots.
h_est, _ = ls_est (y, no)
h_est = h_est[0,0,0,0,0,0]

plt.figure()
plt.plot(np.real(h_perf.cpu().numpy()))
plt.plot(np.imag(h_perf.cpu().numpy()))
plt.plot(np.real(h_est.cpu().numpy()), "--")
plt.plot(np.imag(h_est.cpu().numpy()), "--")
plt.xlabel("Subcarrier index")
plt.ylabel("Channel frequency response")
plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
plt.title("Comparison of channel frequency responses");

direction = "downlink"
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)

perfect_csi = True # Change to switch between perfect and imperfect CSI
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)

b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
c = encoder(b)
x = mapper(c)
x_rg = rg_mapper(x)
cir = cdl(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)

# Precode the transmit signal in the frequency domain
# It is here assumed that the transmitter has perfect knowledge of the channel
# One could here reduce this to perfect knowledge of the channel for the first
# OFDM symbol, or a noisy version of it to take outdated transmit CSI into account.
# `g` is the post-beamforming or `effective channel` that can be
# used to simulate perfect CSI at the receiver.
x_rg, g = zf_precoder(x_rg, h_freq)

y = channel_freq(x_rg, h_freq, no)

if perfect_csi:
    # The receiver gets here the effective channel after precoding as CSI
    h_hat, err_var = g, 0.
else:
    h_hat, err_var = ls_est (y, no)

x_hat, no_eff = lmmse_equ(y, h_hat, err_var, no)
llr = demapper(x_hat, no_eff)
b_hat = decoder(llr)
ber = compute_ber(b, b_hat)
print("BER: {}".format(ber))

def fun(cdl_model):
    """Generates a histogram of the channel condition numbers"""

    # Setup a CIR generator
    cdl = CDL(cdl_model, delay_spread, carrier_frequency,
              ut_array, bs_array, "uplink", min_speed=0)

    # Generate random CIR realizations
    # As we need only a single sample in time, the sampling_frequency
    # does not matter.
    cir = cdl(2000, 1, 1)

    # Compute the frequency response
    h = cir_to_ofdm_channel(frequencies, *cir, normalize=True)

    # Reshape to [batch_size, fft_size, num_rx_ant, num_tx_ant]
    h = torch.squeeze(h)
    h = torch.permute(h, (0, 3, 1, 2))

    # Compute condition number
    c = np.reshape(np.linalg.cond(h.cpu().numpy()), [-1])

    # Compute normalized histogram
    hist, bins = np.histogram(c, 150, (1, 150))
    hist = hist / np.sum(hist)
    return bins[:-1], hist

plt.figure()
for cdl_model in ["A", "B", "C", "D", "E"]:
    bins, hist = fun(cdl_model)
    plt.plot(bins, np.cumsum(hist))
plt.xlim([0,150])
plt.legend(["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"])
plt.xlabel("Channel Condition Number")
plt.ylabel("CDF")
plt.title("CDF of the condition number of 8x4 MIMO channels")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

class Model(Block):
    """This block simulates OFDM MIMO transmissions over the CDL model.

    Simulates point-to-point transmissions between a UT and a BS.
    Uplink and downlink transmissions can be realized with either perfect CSI
    or channel estimation. ZF Precoding for downlink transmissions is assumed.
    The receiver (in both uplink and downlink) applies LS channel estimation
    and LMMSE MIMO equalization. A 5G LDPC code as well as QAM modulation are
    used.

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.
        Time-domain simulations are generally slower and consume more memory.
        They allow modeling of inter-symbol interference and channel changes
        during the duration of an OFDM symbol.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use. Note that "D" and "E" are LOS models that are
        not well suited for the transmissions of multiple streams.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed. For downlink
        transmissions, the transmitter is always assumed to have perfect CSI.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of independent Mote Carlo simulations
        to be performed at once. The larger this number, the larger the memory
        requiremens.

    ebno_db : float
        The Eb/No [dB]. This value is converted to an equivalent noise power
        by taking the modulation order, coderate, pilot and OFDM-related
        overheads into account.

    Output
    ------
    b : [batch_size, 1, num_streams, k], torch.float
        The tensor of transmitted information bits for each stream.

    b_hat : [batch_size, 1, num_streams, k], torch.float
        The tensor of received information bits for each stream.
    """

    def __init__(self,
                 domain,
                 direction,
                 cdl_model,
                 delay_spread,
                 perfect_csi,
                 speed,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 15e3
                ):
        super().__init__()

        # Provided parameters
        self._domain = domain
        self._direction = direction
        self._cdl_model = cdl_model
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 72
        self._num_ofdm_symbols = 14
        self._num_ut_ant = 4 # Must be a multiple of two as dual-polarized antennas are used
        self._num_bs_ant = 8 # Must be a multiple of two as dual-polarized antennas are used
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 2
        self._coderate = 0.5

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        self._ut_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_ut_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=True)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        if self._domain == "time":
            # Time-domain simulations

            a, tau = self._cdl(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=True)

            # As precoding is done in the frequency domain, we need to downsample
            # the path gains `a` to the OFDM symbol rate prior to converting the CIR
            # to the channel frequency response.
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            x_time = self._modulator(x_rg)
            y_time = self._channel_time(x_time, h_time, no)

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations

            cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1/self._rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            y = self._channel_freq(x_rg, h_freq, no)

        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_scs(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est (y, no)

        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        b_hat = self._decoder(llr)

        return b, b_hat

torch._dynamo.reset()

UL_SIMS = {
    "ebno_db" : list(np.arange(-5, 20, 4.0)),
    "cdl_model" : ["A", "B", "C", "D", "E"],
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "uplink",
    "perfect_csi" : True,
    "speed" : 0.0,
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for cdl_model in UL_SIMS["cdl_model"]:

    model = Model(domain=UL_SIMS["domain"],
                  direction=UL_SIMS["direction"],
                  cdl_model=cdl_model,
                  delay_spread=UL_SIMS["delay_spread"],
                  perfect_csi=UL_SIMS["perfect_csi"],
                  speed=UL_SIMS["speed"],
                  cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])

    ber, bler = sim_ber(model,
                        UL_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000,
                        target_bler=1e-3,
                        compile_mode="reduce-overhead")

    UL_SIMS["ber"].append(list(ber.cpu().numpy()))
    UL_SIMS["bler"].append(list(bler.cpu().numpy()))

UL_SIMS["duration"] = time.time() - start

print("Simulation duration: {:1.2f} [h]".format(UL_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("8x4 MIMO Uplink - Frequency Domain Modeling");
plt.ylim([1e-3, 1.1])
legend = []
for i, bler in enumerate(UL_SIMS["bler"]):
    plt.semilogy(UL_SIMS["ebno_db"], bler)
    legend.append("CDL-{}".format(UL_SIMS["cdl_model"][i]))
plt.legend(legend);

torch._dynamo.reset()

DL_SIMS = {
    "ebno_db" : list(np.arange(-5, 20, 4.0)),
    "cdl_model" : ["A", "B", "C", "D", "E"],
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "downlink",
    "perfect_csi" : True,
    "speed" : 0.0,
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for cdl_model in DL_SIMS["cdl_model"]:

    model = Model(domain=DL_SIMS["domain"],
                  direction=DL_SIMS["direction"],
                  cdl_model=cdl_model,
                  delay_spread=DL_SIMS["delay_spread"],
                  perfect_csi=DL_SIMS["perfect_csi"],
                  speed=DL_SIMS["speed"],
                  cyclic_prefix_length=DL_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=DL_SIMS["pilot_ofdm_symbol_indices"])

    ber, bler = sim_ber(model,
                        DL_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=100,
                        target_bler=1e-3,
                        compile_mode="reduce-overhead")

    DL_SIMS["ber"].append(list(ber.cpu().numpy()))
    DL_SIMS["bler"].append(list(bler.cpu().numpy()))

DL_SIMS["duration"] = time.time() -  start

print("Simulation duration: {:1.2f} [h]".format(DL_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("8x4 MIMO Downlink - Frequency Domain Modeling");
plt.ylim([1e-3, 1.1])
legend = []
for i, bler in enumerate(DL_SIMS["bler"]):
    plt.semilogy(DL_SIMS["ebno_db"], bler)
    legend.append("CDL-{}".format(DL_SIMS["cdl_model"][i]))
plt.legend(legend);

torch._dynamo.reset()

MOBILITY_SIMS = {
    "ebno_db" : list(np.arange(0, 32, 2.0)),
    "cdl_model" : "D",
    "delay_spread" : 100e-9,
    "domain" : "freq",
    "direction" : "uplink",
    "perfect_csi" : [True, False],
    "speed" : [0.0, 20.0],
    "cyclic_prefix_length" : 6,
    "pilot_ofdm_symbol_indices" : [0],
    "ber" : [],
    "bler" : [],
    "duration" : None
}

start = time.time()

for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
    for speed in MOBILITY_SIMS["speed"]:

        model = Model(domain=MOBILITY_SIMS["domain"],
                  direction=MOBILITY_SIMS["direction"],
                  cdl_model=MOBILITY_SIMS["cdl_model"],
                  delay_spread=MOBILITY_SIMS["delay_spread"],
                  perfect_csi=perfect_csi,
                  speed=speed,
                  cyclic_prefix_length=MOBILITY_SIMS["cyclic_prefix_length"],
                  pilot_ofdm_symbol_indices=MOBILITY_SIMS["pilot_ofdm_symbol_indices"])

        ber, bler = sim_ber(model,
                        MOBILITY_SIMS["ebno_db"],
                        batch_size=256,
                        max_mc_iter=100,
                        num_target_block_errors=1000,
                        target_bler=1e-3,
                        compile_mode="reduce-overhead")

        MOBILITY_SIMS["ber"].append(list(ber.cpu().numpy()))
        MOBILITY_SIMS["bler"].append(list(bler.cpu().numpy()))

MOBILITY_SIMS["duration"] = time.time() - start

print("Simulation duration: {:1.2f} [h]".format(MOBILITY_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("CDL-D MIMO Uplink - Impact of UT mobility")

i = 0
for perfect_csi in MOBILITY_SIMS["perfect_csi"]:
    for speed in MOBILITY_SIMS["speed"]:
        style = "{}".format("-" if perfect_csi else "--")
        s = "{} CSI {}[m/s]".format("Perf." if perfect_csi else "Imperf.", speed)
        plt.semilogy(MOBILITY_SIMS["ebno_db"],
                     MOBILITY_SIMS["bler"][i],
                      style, label=s,)
        i += 1
plt.legend();
plt.ylim([1e-3, 1]);

torch._dynamo.reset()

CP_SIMS = {
    "ebno_db" : list(np.arange(0, 17, 2.0)),
    "cdl_model" : "C",
    "delay_spread" : 100e-9,
    "subcarrier_spacing" : 15e3,
    "domain" : ["freq", "time"],
    "direction" : "uplink",
    "perfect_csi" : False,
    "speed" : 3.0,
    "cyclic_prefix_length" : [20, 2],
    "pilot_ofdm_symbol_indices" : [2, 11],
    "ber" : [],
    "bler" : [],
    "duration": None
}

start = time.time()

for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
    for domain in CP_SIMS["domain"]:
        model = Model(domain=domain,
                  direction=CP_SIMS["direction"],
                  cdl_model=CP_SIMS["cdl_model"],
                  delay_spread=CP_SIMS["delay_spread"],
                  perfect_csi=CP_SIMS["perfect_csi"],
                  speed=CP_SIMS["speed"],
                  cyclic_prefix_length=cyclic_prefix_length,
                  pilot_ofdm_symbol_indices=CP_SIMS["pilot_ofdm_symbol_indices"],
                  subcarrier_spacing=CP_SIMS["subcarrier_spacing"])

        ber, bler = sim_ber(model,
                        CP_SIMS["ebno_db"],
                        batch_size=64,
                        max_mc_iter=1000,
                        num_target_block_errors=1000,
                        target_bler=1e-3,
                        compile_mode="reduce-overhead")

        CP_SIMS["ber"].append(list(ber.cpu().numpy()))
        CP_SIMS["bler"].append(list(bler.cpu().numpy()))

CP_SIMS["duration"] = time.time() - start

print("Simulation duration: {:1.2f} [h]".format(CP_SIMS["duration"]/3600))

plt.figure()
plt.xlabel(r"$E_b/N_0$ (dB)")
plt.ylabel("BLER")
plt.grid(which="both")
plt.title("CDL-C MIMO Uplink - Impact of Cyclic Prefix Length")

i = 0
for cyclic_prefix_length in CP_SIMS["cyclic_prefix_length"]:
    for domain in CP_SIMS["domain"]:
        s = "{} Domain, CP length: {}".format("Freq" if domain=="freq" else "Time",
                                               cyclic_prefix_length)
        plt.semilogy(CP_SIMS["ebno_db"],
                     CP_SIMS["bler"][i],
                     label=s)
        i += 1
plt.legend();
plt.ylim([1e-3, 1]);

