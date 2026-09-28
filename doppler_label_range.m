function fdClip = doppler_label_range(cfg)
%DOPPLER_LABEL_RANGE [lo hi] range (Hz) that every Doppler label lies in.
%   For cfg.labelSampling = 'balanced' this is
%   [-(qMax + 0.5), qMax + 0.5) subcarriers, so round-half-up decomposition
%   eps = q + delta always gives q in {-qMax, ..., qMax}, delta in [-0.5, 0.5).
%   Consumes no random numbers.

    switch cfg.labelSampling
        case 'balanced'
            edge   = (cfg.qMax + 0.5) * cfg.subcarrierSpacing;
            % Upper end kept ~0.05 Hz below the edge so it stays below
            % qMax + 0.5 even after the float32 cast of the stored label.
            fdClip = [-edge, edge * (1 - 1e-6)];
        case 'uniform'
            fdClip = cfg.fdRange;
        otherwise
            error('doppler_label_range:labelSampling', ...
                  'Unknown labelSampling ''%s''.', cfg.labelSampling);
    end
end
