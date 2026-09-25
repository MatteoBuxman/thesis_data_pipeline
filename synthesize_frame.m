function [rx, tx, fs] = synthesize_frame(rmc, fdHz, phaseRad, snrDb)
%SYNTHESIZE_FRAME One LTE downlink frame through the Doppler + AWGN channel.
%   [rx, tx, fs] = SYNTHESIZE_FRAME(rmc, fdHz, phaseRad, snrDb) generates a
%   10 ms reference waveform from the RMC config rmc, normalises it to unit
%   average power, applies a constant frequency offset fdHz with initial
%   carrier phase phaseRad, and adds complex white Gaussian noise at snrDb.
%   Pass snrDb = Inf for a noiseless frame.
%
%   rx  - received waveform (column vector, complex double)
%   tx  - normalised transmitted waveform, before the channel
%   fs  - sample rate (Hz)

    bits = randi([0 1], sum(rmc.PDSCH.TrBlkSizes), 1);
    [tx, ~, info] = lteRMCDLTool(rmc, {bits});
    fs = info.SamplingRate;

    tx = tx / sqrt(mean(abs(tx).^2));

    n  = (0:numel(tx)-1).';
    rx = tx .* exp(1j * (2*pi*fdHz*n/fs + phaseRad));

    if isfinite(snrDb)
        noiseVar = 10^(-snrDb/10);         % signal power is 1
        rx = rx + sqrt(noiseVar/2) * complex(randn(size(rx)), randn(size(rx)));
    end
end
