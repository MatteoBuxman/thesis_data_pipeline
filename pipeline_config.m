function cfg = pipeline_config()
%PIPELINE_CONFIG Parameters for the LTE Doppler dataset.
%   cfg = PIPELINE_CONFIG() returns the struct consumed by GENERATE_DATASET.
%   Edit values here rather than in the generator so every dataset is
%   reproducible from (config, seed).

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

    % --- Channel --------------------------------------------------------
    % Max Doppler at 2 GHz, 550 km altitude is ~46.6 kHz (satellite at the
    % horizon); the range below adds margin. Doppler rate (<~700 Hz/s) changes
    % fd by <10 Hz over a 10 ms frame, so fd is held constant per frame.
    cfg.fc          = 2e9;                 % carrier frequency (Hz)
    cfg.altitude    = 550e3;               % orbit altitude (m), documentation only
    cfg.fdRange     = [-50e3 50e3];        % Doppler label range (Hz), uniform
    cfg.snrRangeDb  = [-5 20];             % SNR range (dB), uniform

    % --- Windowing and size ---------------------------------------------
    cfg.nFrames         = 20000;
    cfg.windowsPerFrame = 10;
    cfg.windowLength    = 256;             % samples per network input
    cfg.splitFractions  = [0.8 0.1 0.1];   % train / val / test, assigned per frame
    cfg.framesPerChunk  = 500;             % frames held in memory before each HDF5 write
end
