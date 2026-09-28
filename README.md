# LTE Doppler dataset pipeline

Generates labelled windows of raw LTE downlink IQ samples with a known Doppler
shift, for training a TCN (Temporal Convolutional Network) Doppler estimator for
LEO direct-to-cell links.

Requires MATLAB with LTE Toolbox (developed on R2025b).

## Signal chain

```
RMC R.4 frame (LTE Toolbox)  →  normalise to unit power  →  Doppler trajectory + carrier phase
  →  AWGN  →  amplitude augmentation  →  256-sample windows  →  per-window labels
```

The dataset recipe follows Rehman et al., *Coarse-to-Fine Doppler Compensation
in 6G VLEO*, IEEE Trans. Commun., 2026, Sec. IV-C. The structure is copied; the
parameters that were tied to their 20 GHz / 300 km VLEO system (or were not
physically possible) are replaced by the equivalent values for this link.
`pipeline_config.m` marks each such line.

1. **Reference waveform.** `lteRMCDLTool` builds a standard-compliant 10 ms
   downlink frame (PSS, SSS, PBCH, CRS, control, PDSCH). OCNG (OFDMA Channel
   Noise Generator) fills the PDSCH and PDCCH resource elements the RMC leaves
   empty, so the cell looks fully loaded. Without PDCCH OCNG, OFDM symbols 1–3
   of subframe 5 are completely silent at 1.4 MHz. Modulation,
   cell ID and system frame number are randomised per frame.
2. **Doppler anchor (class-balanced).** The normalised CFO is
   `ε = fd / Δf = q + δ` with Δf = 15 kHz. As in the paper, `q` is drawn
   uniformly from the integer classes and `δ` uniformly from [−0.5, 0.5).
   Max Doppler at 2 GHz / 550 km is ~46.6 kHz ≈ 3.1 subcarriers, so there are
   7 classes, q ∈ {−3, …, 3} (the paper has 33 at 20 GHz). Drawing fd uniformly
   in Hz instead would under-represent the outer classes (12.5% vs 15%).
3. **Doppler trajectory.** Each frame's anchor (the Doppler at the frame
   centre) is turned into a per-sample instantaneous frequency by one of the
   paper's four families, in the paper's proportions:

   | Family | Share | Paper parameters | Used here |
   |---|---|---|---|
   | Linear sweep | 40% | ±50 Hz/ms (50 kHz/s) | ±700 Hz/s, the physical max at 2 GHz / 550 km |
   | Sinusoidal | 30% | period 10–300 s, amplitude ≤ 15 subcarriers | same periods; amplitude capped so the peak rate ≤ 700 Hz/s |
   | Step (handover) | 15% | jump of 5–20 subcarriers | new Doppler redrawn from the label distribution, ≥ 1 subcarrier away |
   | Random walk | 15% | σ = 0.1 subcarrier per symbol (~21 MHz/s) | σ = 10 Hz per OFDM symbol (oscillator-drift scale) |

   Within one 10 ms frame the smooth families are nearly linear (a few Hz of
   change), so most of the within-window variation comes from steps and
   random walks. The carrier phase is the running integral of the frequency:
   `rx[n] = tx[n] · exp(j(φ + 2π/fs · Σ_{m<n} fd[m]))`.
4. **Noise.** Complex white Gaussian noise with variance `10^(−SNR/10)` relative
   to the unit-power signal, SNR uniform on 0–20 dB (as in the paper).
5. **Augmentation.** 15% of frames are scaled by a factor in 0.8–1.2 (signal
   and noise together, so the SNR label is unchanged). The paper's random phase
   rotation and ±5-symbol temporal shifting are already covered: every frame
   has a uniform random carrier phase, and every window starts at a random
   sample.
6. **Windows and labels.** `windowsPerFrame` windows of `windowLength` samples
   at random start positions. A window's `fd_hz` label is the mean
   instantaneous frequency over that window (equivalently, the slope of its
   carrier phase), with `eps`, `q_int` and `delta` derived from it. Windows
   that straddle a handover step are flagged in `step_in_window`; their label
   is a mix of two Dopplers, so decide whether to train on them.
7. **Splits.** 100,000 / 20,000 / 50,000 train / val / test windows, as in the
   paper, assigned per frame so no frame spans two sets.

## Default parameters (`pipeline_config.m`)

| Parameter | Value |
|---|---|
| Bandwidth | 1.4 MHz (6 RB), fs = 1.92 MHz, FFT size 128 |
| Carrier | 2 GHz (max physical Doppler ≈ ±46.6 kHz at 550 km) |
| Doppler label | class-balanced: q ∈ {−3..3} uniform, δ ∈ [−0.5, 0.5) uniform, fd = (q+δ)·15 kHz |
| Doppler trajectories | 40% linear, 30% sinusoidal, 15% step, 15% random walk |
| SNR | uniform 0 to 20 dB |
| Amplitude augmentation | 15% of frames, factor 0.8–1.2 |
| Modulation | 60% QPSK, 30% 16QAM, 10% 64QAM |
| Frames × windows | 17,000 × 10 = 170,000 windows of 256 samples |
| Split | 100k / 20k / 50k train / val / test, assigned per frame |

## Usage

```bash
matlab -batch test_pipeline       # sanity checks (a few minutes)
matlab -batch generate_dataset    # full dataset → data/lte_doppler_1p4MHz.h5
```

## Output format (HDF5)

Read with `h5py`. `iq` has shape `(N, 256, 2)`, channel 0 = I, 1 = Q.

| Dataset | Type | Meaning |
|---|---|---|
| `iq` | float32 | Received IQ samples |
| `fd_hz` | float32 | Doppler label: mean over the window (Hz) |
| `eps` | float32 | Normalised CFO, fd / 15 kHz |
| `q_int` | int8 | Integer CFO class, round-half-up of eps |
| `delta` | float32 | Fractional CFO, eps − q, in [−0.5, 0.5) |
| `fd_rate_hz_s` | float32 | Doppler change across the window (Hz/s) |
| `trajectory` | uint8 | Index into the `trajectory_types` file attribute |
| `step_in_window` | uint8 | 1 if a handover step falls inside the window |
| `snr_db` | float32 | SNR (dB) |
| `amp_scale` | float32 | Amplitude augmentation factor (1 if none) |
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
    q = f["q_int"][train]       # integer head target (class index = q + q_max)
    d = f["delta"][train]       # fractional head target
```

## Known limitations

- AWGN only; no multipath or LEO-specific (e.g. Rician) fading yet.
- A real UE's oscillator error (up to ±0.1 ppm, ≈ ±200 Hz at 2 GHz) is
  indistinguishable from Doppler, so in practice the estimator targets the total
  frequency offset.
