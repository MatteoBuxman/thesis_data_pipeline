function cfg = pipeline_config()
%PIPELINE_CONFIG Parameters for the LTE Doppler dataset.
%   cfg = PIPELINE_CONFIG() returns the struct consumed by GENERATE_DATASET.
%   Edit values here rather than in the generator so every dataset is
%   reproducible from (config, seed).
%
%   The dataset recipe follows Rehman et al., "Coarse-to-Fine Doppler
%   Compensation in 6G VLEO", IEEE Trans. Commun., 2026, Sec. IV-C:
%   class-balanced integer CFO with uniform fractional CFO, a fixed mix of
%   Doppler trajectory families, uniform SNR, light augmentation, and
%   100k / 20k / 50k train / val / test windows. Where the paper's numbers
%   are tied to their 20 GHz / 300 km VLEO system, the equivalent value for
%   this 2 GHz / 550 km LTE link is used instead; each such line says so.

    % --- Reproducibility and output -------------------------------------
    cfg.seed    = 42;
    cfg.outFile = fullfile(fileparts(mfilename('fullpath')), 'data', ...
                           'lte_doppler_1p4MHz.h5');

    % --- Waveform -------------------------------------------------------
    % RMC R.4: 1.4 MHz (6 RB), sample rate 1.92 MHz, FFT size 128.
    cfg.rc          = 'R.4';
    cfg.modulations = {'QPSK', '16QAM', '64QAM'};
    cfg.modulationProbs = [0.6 0.3 0.1];   % LEO links are mostly QPSK
    cfg.ocngProb    = 1.0;                 % fraction of frames with OCNG fill (fully loaded cell)
    cfg.subcarrierSpacing = 15e3;          % Hz; normalised CFO eps = fd / subcarrierSpacing
    cfg.symbolsPerFrame   = 140;           % OFDM symbols per 10 ms frame (normal CP)

    % --- Doppler labels -------------------------------------------------
    % Paper: q uniform over 33 classes {-16..16}, delta uniform [-0.5, 0.5).
    % Here: max Doppler at 2 GHz / 550 km is ~46.6 kHz = 3.1 subcarriers, so
    % the same construction gives 7 classes {-3..3}, i.e. |fd| < 52.5 kHz.
    cfg.fc            = 2e9;               % carrier frequency (Hz)
    cfg.altitude      = 550e3;             % orbit altitude (m), documentation only
    cfg.labelSampling = 'balanced';        % 'balanced' (paper) or 'uniform' (fdRange)
    cfg.qMax          = 3;                 % integer classes -qMax..qMax
    cfg.fdRange       = [-50e3 50e3];      % only used when labelSampling = 'uniform'

    % --- Doppler trajectories (paper Sec. IV-C, items 1-4) ----------------
    % Paper mix: 40% linear sweeps, 30% sinusoidal, 15% steps, 15% random walks.
    cfg.trajectoryTypes = {'constant', 'linear', 'sinusoid', 'step', 'randomwalk'};
    cfg.trajectoryProbs = [0           0.40      0.30        0.15    0.15];
    % Linear: paper uses +-50 Hz/ms (50 kHz/s), which exceeds the physical
    % Doppler rate even for their own 20 GHz VLEO case (~13 kHz/s). The
    % physical maximum here is ~700 Hz/s.
    cfg.fdRateMax      = 700;              % Hz/s
    % Sinusoidal: paper periods 10-300 s, amplitude up to 15 subcarriers;
    % amplitude is capped so the peak rate stays <= fdRateMax.
    cfg.sinPeriodRange = [10 300];         % s
    cfg.sinAmpMaxHz    = cfg.qMax * cfg.subcarrierSpacing;
    % Steps (handover): paper jumps 5-20 subcarriers, larger than this whole
    % label range; here the new Doppler is redrawn from the label
    % distribution, at least 1 subcarrier away.
    cfg.stepMinHz      = cfg.subcarrierSpacing;
    % Random walk: paper uses sigma = 0.1 subcarriers per symbol (1.5 kHz per
    % symbol, i.e. ~21 MHz/s, not a Doppler process). Here sigma is set to an
    % oscillator-drift scale: ~120 Hz spread over a 140-symbol frame.
    cfg.rwSigmaHz      = 10;               % Hz per OFDM symbol

    % --- Noise ----------------------------------------------------------
    cfg.snrRangeDb  = [0 20];              % SNR range (dB), uniform (paper: 0-20 dB)

    % --- Augmentation (paper Sec. IV-C) -----------------------------------
    % Random phase rotation: every frame already gets a uniform carrier phase.
    % Temporal shifting: every window already starts at a random sample.
    cfg.ampScaleProb  = 0.15;              % fraction of frames scaled
    cfg.ampScaleRange = [0.8 1.2];         % amplitude factor (signal and noise together)

    % --- Windowing and size ---------------------------------------------
    % Paper: 100,000 train / 20,000 val / 50,000 test. 17,000 frames x 10
    % windows = 170,000 windows, split per frame so no frame spans two sets.
    cfg.nFrames         = 17000;
    cfg.windowsPerFrame = 10;
    cfg.windowLength    = 256;             % samples per network input (paper L = 256)
    cfg.splitFractions  = [100 20 50] / 170;   % train / val / test, assigned per frame
    cfg.framesPerChunk  = 500;             % frames held in memory before each HDF5 write
end
