function [rx, tx, fs] = synthesize_frame(rmc, fdHz, phaseRad, snrDb)
%SYNTHESIZE_FRAME One LTE downlink frame through the Doppler + AWGN channel.
%   [rx, tx, fs] = SYNTHESIZE_FRAME(rmc, fdHz, phaseRad, snrDb) generates a
%   10 ms reference waveform from the RMC config rmc, normalises it to unit
%   average power, applies the frequency offset fdHz with initial carrier
%   phase phaseRad, and adds complex white Gaussian noise at snrDb.
%   Pass snrDb = Inf for a noiseless frame.
%
%   fdHz is either a scalar (constant offset) or a vector with one
%   instantaneous frequency per sample (see DOPPLER_TRAJECTORY). For a vector
%   the phase is the running integral of the frequency:
%       phase[n] = phaseRad + 2*pi/fs * sum_{m<n} fdHz[m]
%   which reduces to phaseRad + 2*pi*fd*n/fs when fdHz is constant.
%
%   rx  - received waveform (column vector, complex double)
%   tx  - normalised transmitted waveform, before the channel
%   fs  - sample rate (Hz)

    bits = randi([0 1], sum(rmc.PDSCH.TrBlkSizes), 1);
    [tx, ~, info] = lteRMCDLTool(rmc, {bits});
    fs = info.SamplingRate;

    tx = tx / sqrt(mean(abs(tx).^2));

    if isscalar(fdHz)
        n     = (0:numel(tx)-1).';
        phase = 2*pi*fdHz*n/fs;
    else
        assert(numel(fdHz) == numel(tx), 'synthesize_frame:length', ...
               'fdHz has %d samples, frame has %d.', numel(fdHz), numel(tx));
        f     = fdHz(:);
        phase = 2*pi/fs * [0; cumsum(f(1:end-1))];
    end
    rx = tx .* exp(1j * (phase + phaseRad));

    if isfinite(snrDb)
        noiseVar = 10^(-snrDb/10);         % signal power is 1
        rx = rx + sqrt(noiseVar/2) * complex(randn(size(rx)), randn(size(rx)));
    end
end
