function [fInst, info] = doppler_trajectory(type, fd0, Ns, fs, cfg)
%DOPPLER_TRAJECTORY Instantaneous Doppler (Hz) for every sample of a frame.
%   [fInst, info] = DOPPLER_TRAJECTORY(type, fd0, Ns, fs, cfg) returns an
%   Ns-by-1 vector of instantaneous frequency offsets. fd0 is the Doppler at
%   the frame centre (the anchor drawn by SAMPLE_DOPPLER).
%
%   The families follow Rehman et al. (2026), Sec. IV-C, but with parameters
%   scaled to what is physically possible for this link (see
%   PIPELINE_CONFIG). Within one 10 ms frame the smooth families (linear,
%   sinusoid) are all close to fd0 + rate * t, because the Doppler rate is at
%   most a few hundred Hz/s; only 'step' and 'randomwalk' change the shape.
%
%   type        behaviour
%   'constant'  fd0 for the whole frame
%   'linear'    fd0 + rate * t, rate ~ U(-fdRateMax, fdRateMax)
%   'sinusoid'  fd0 + A (sin(2 pi t / T + psi) - sin(psi)), T ~ U(sinPeriodRange),
%               A capped so the peak rate 2 pi A / T <= fdRateMax
%   'step'      fd0 before a random sample, fd1 after (beam/satellite handover);
%               fd1 drawn like fd0, at least stepMinHz away
%   'randomwalk' fd0 plus a Gaussian random walk, one increment of std
%               rwSigmaHz per OFDM symbol, zero at the frame centre
%
%   info.stepIndex is the 1-based first sample after the step (NaN otherwise).
%   Every trajectory is clipped to DOPPLER_LABEL_RANGE(cfg).
%
%   Pure MATLAB, no toolbox needed.

    t = ((0:Ns-1).' - (Ns-1)/2) / fs;     % seconds, zero at the frame centre
    info = struct('stepIndex', NaN, 'rateHz_s', 0);

    switch type
        case 'constant'
            fInst = fd0 * ones(Ns, 1);

        case 'linear'
            rate  = cfg.fdRateMax * (2*rand - 1);
            fInst = fd0 + rate * t;
            info.rateHz_s = rate;

        case 'sinusoid'
            T    = cfg.sinPeriodRange(1) + diff(cfg.sinPeriodRange) * rand;
            Amax = min(cfg.sinAmpMaxHz, cfg.fdRateMax * T / (2*pi));
            A    = Amax * rand;
            psi  = 2*pi * rand;
            fInst = fd0 + A * (sin(2*pi*t/T + psi) - sin(psi));
            info.rateHz_s = 2*pi*A/T * cos(psi);

        case 'step'
            fd1 = fd0;
            for tries = 1:100
                fd1 = sample_doppler(cfg);
                if abs(fd1 - fd0) >= cfg.stepMinHz
                    break
                end
            end
            k = randi([2, Ns]);               % first sample at the new Doppler
            fInst = [fd0 * ones(k-1, 1); fd1 * ones(Ns-k+1, 1)];
            info.stepIndex = k;

        case 'randomwalk'
            symLen = Ns / cfg.symbolsPerFrame;
            sym    = floor((0:Ns-1).' / symLen) + 1;           % symbol of each sample
            walk   = cumsum(cfg.rwSigmaHz * randn(cfg.symbolsPerFrame, 1));
            walk   = walk - walk(round(cfg.symbolsPerFrame / 2)); % zero at the centre
            fInst  = fd0 + walk(sym);

        otherwise
            error('doppler_trajectory:type', 'Unknown trajectory type ''%s''.', type);
    end

    fdClip = doppler_label_range(cfg);
    fInst  = min(max(fInst, fdClip(1)), fdClip(2));
end
