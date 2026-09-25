# LTE Doppler dataset pipeline

Generates labelled windows of raw LTE downlink IQ samples with a known Doppler
shift, for training a TCN (Temporal Convolutional Network) Doppler estimator for
LEO direct-to-cell links.

Requires MATLAB with LTE Toolbox (developed on R2025b).

## Signal chain

```
RMC R.4 frame (LTE Toolbox)  →  normalise to unit power  →  Doppler + carrier phase  →  AWGN  →  256-sample windows
```

1. **Reference waveform.** `lteRMCDLTool` builds a standard-compliant 10 ms
   downlink frame (PSS, SSS, PBCH, CRS, control, PDSCH). OCNG (OFDMA Channel
   Noise Generator) fills the PDSCH and PDCCH resource elements the RMC leaves
   empty, so the cell looks fully loaded. Without PDCCH OCNG, OFDM symbols 1–3
   of subframe 5 are completely silent at 1.4 MHz. Modulation,
   cell ID and system frame number are randomised per frame.
2. **Doppler.** A constant frequency offset `fd` and random initial phase:
   `rx[n] = tx[n] · exp(j(2π·fd·n/fs + φ))`. Holding `fd` constant per frame is
   justified because the Doppler rate at 2 GHz / 550 km is below ~700 Hz/s,
   which moves `fd` by under 10 Hz across a frame.
3. **Noise.** Complex white Gaussian noise with variance `10^(−SNR/10)` relative
   to the unit-power signal.
4. **Windows.** `windowsPerFrame` windows of `windowLength` samples at random
   start positions.

## Default parameters (`pipeline_config.m`)

| Parameter | Value |
|---|---|
| Bandwidth | 1.4 MHz (6 RB), fs = 1.92 MHz, FFT size 128 |
| Carrier | 2 GHz (max physical Doppler ≈ ±46.6 kHz at 550 km) |
| Doppler label | uniform ±50 kHz |
| SNR | uniform −5 to 20 dB |
| Modulation | 60% QPSK, 30% 16QAM, 10% 64QAM |
| Frames × windows | 20,000 × 10 = 200,000 windows of 256 samples |
| Split | 80 / 10 / 10 train / val / test, assigned per frame |

## Usage

```bash
matlab -batch test_pipeline       # sanity checks (~1 min)
matlab -batch generate_dataset    # full dataset → data/lte_doppler_1p4MHz.h5
```

## Output format (HDF5)

Read with `h5py`. `iq` has shape `(N, 256, 2)`, channel 0 = I, 1 = Q.

| Dataset | Type | Meaning |
|---|---|---|
| `iq` | float32 | Received IQ samples |
| `fd_hz` | float32 | Doppler label (Hz) |
| `snr_db` | float32 | SNR (dB) |
| `phase_rad` | float32 | Carrier phase at frame start |
| `start_sample` | int32 | 0-based window start within its frame |
| `frame_index` | int32 | 0-based source frame |
| `split` | uint8 | 0 train, 1 val, 2 test |
| `ncellid` | int16 | Physical cell ID |
| `nframe` | int16 | System frame number |
| `modulation` | uint8 | Index into the `modulations` file attribute |
| `ocng` | uint8 | 1 if OCNG filled unused resource elements |

Root attributes record the sample rate, ranges, seed and MATLAB version.

The subframe a window starts in is `start_sample // 1920`. Each slot is 960
samples: OFDM symbols of 128 samples, with a 10-sample cyclic prefix on the
first symbol and 9 on the other six.

```python
import h5py
with h5py.File("data/lte_doppler_1p4MHz.h5", "r") as f:
    train = f["split"][:] == 0
    x = f["iq"][train]          # (N_train, 256, 2)
    y = f["fd_hz"][train]
```

## Known limitations

- AWGN only; no multipath or LEO-specific (e.g. Rician) fading yet.
- A real UE's oscillator error (up to ±0.1 ppm, ≈ ±200 Hz at 2 GHz) is
  indistinguishable from Doppler, so in practice the estimator targets the total
  frequency offset.
