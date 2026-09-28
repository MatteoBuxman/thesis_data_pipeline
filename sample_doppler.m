function fd = sample_doppler(cfg)
%SAMPLE_DOPPLER Draw one anchor Doppler value (Hz) according to cfg.
%
%   cfg.labelSampling = 'balanced' (Rehman et al. 2026, Sec. IV-C):
%       q ~ uniform integer in {-qMax, ..., qMax}
%       delta ~ uniform in [-0.5, 0.5)
%       fd = (q + delta) * subcarrierSpacing
%     Every integer class then gets the same number of frames, which uniform
%     sampling of fd in Hz does not give (the outer classes are truncated).
%
%   cfg.labelSampling = 'uniform':
%       fd ~ uniform on cfg.fdRange (Hz).
%
%   The label range itself is DOPPLER_LABEL_RANGE(cfg). Pure MATLAB.

    switch cfg.labelSampling
        case 'balanced'
            q     = randi([-cfg.qMax, cfg.qMax]);
            delta = rand - 0.5;
            fd    = (q + delta) * cfg.subcarrierSpacing;
        case 'uniform'
            fd = cfg.fdRange(1) + diff(cfg.fdRange) * rand;
        otherwise
            error('sample_doppler:labelSampling', ...
                  'Unknown labelSampling ''%s''.', cfg.labelSampling);
    end
end
