function generate_dataset(cfg)
%GENERATE_DATASET Build the LTE Doppler training set and write it to HDF5.
%   GENERATE_DATASET() uses PIPELINE_CONFIG. GENERATE_DATASET(cfg) uses cfg.
%
%   Each frame gets random modulation, cell ID, frame number, Doppler, carrier
%   phase and SNR. cfg.windowsPerFrame windows of cfg.windowLength samples are
%   cut from each frame at random start positions.
%
%   HDF5 layout (N = nFrames * windowsPerFrame windows). h5py reads the
%   MATLAB [2 L N] array as shape (N, L, 2), channel 0 = I, 1 = Q.
%     /iq            single [2 L N]  received IQ samples
%     /fd_hz         single [N]      Doppler label (Hz)
%     /snr_db        single [N]      SNR (dB)
%     /phase_rad     single [N]      carrier phase at frame start (rad)
%     /start_sample  int32  [N]      0-based window start within its frame
%     /frame_index   int32  [N]      0-based frame the window came from
%     /split         uint8  [N]      0 = train, 1 = val, 2 = test (per frame)
%     /ncellid       int16  [N]      physical cell ID
%     /nframe        int16  [N]      system frame number
%     /modulation    uint8  [N]      index into the 'modulations' attribute (0-based)
%     /ocng          uint8  [N]      1 if OCNG filled the unused PDSCH and PDCCH REs

    if nargin < 1
        cfg = pipeline_config();
    end
    rng(cfg.seed, 'twister');

    % One base config per modulation; lteRMCDL recomputes the dependent fields.
    base = cell(1, numel(cfg.modulations));
    for m = 1:numel(cfg.modulations)
        rmc = lteRMCDL(cfg.rc);
        rmc.TotSubframes = 10;
        rmc.PDSCH.Modulation = cfg.modulations{m};
        base{m} = lteRMCDL(rmc);
    end
    ofdm = lteOFDMInfo(base{1});
    fs   = double(ofdm.SamplingRate);
    Ns   = round(10e-3 * fs);              % samples per frame

    L = cfg.windowLength;
    W = cfg.windowsPerFrame;
    F = cfg.nFrames;
    N = F * W;

    split = assign_splits(F, cfg.splitFractions);
    modCdf = cumsum(cfg.modulationProbs) / sum(cfg.modulationProbs);

    out = cfg.outFile;
    create_file(out, L, N);

    t0 = tic;
    for f0 = 1:cfg.framesPerChunk:F
        f1 = min(f0 + cfg.framesPerChunk - 1, F);
        nw = (f1 - f0 + 1) * W;

        iq    = zeros(2, L, nw, 'single');
        lab.fd_hz        = zeros(nw, 1, 'single');
        lab.snr_db       = zeros(nw, 1, 'single');
        lab.phase_rad    = zeros(nw, 1, 'single');
        lab.start_sample = zeros(nw, 1, 'int32');
        lab.frame_index  = zeros(nw, 1, 'int32');
        lab.split        = zeros(nw, 1, 'uint8');
        lab.ncellid      = zeros(nw, 1, 'int16');
        lab.nframe       = zeros(nw, 1, 'int16');
        lab.modulation   = zeros(nw, 1, 'uint8');
        lab.ocng         = zeros(nw, 1, 'uint8');

        k = 0;
        for f = f0:f1
            m   = find(rand <= modCdf, 1);
            rmc = base{m};
            rmc.NCellID = randi([0 503]);
            rmc.NFrame  = randi([0 1023]);
            useOcng = rand < cfg.ocngProb;
            % PDSCH OCNG fills unscheduled data REs; PDCCH OCNG fills the
            % control region, which is otherwise silent in subframe 5.
            rmc.OCNGPDSCHEnable = onoff(useOcng);
            rmc.OCNGPDCCHEnable = onoff(useOcng);

            fd  = cfg.fdRange(1)    + diff(cfg.fdRange)    * rand;
            snr = cfg.snrRangeDb(1) + diff(cfg.snrRangeDb) * rand;
            phi = 2*pi * rand;

            rx = synthesize_frame(rmc, fd, phi, snr);

            starts = randi([1, Ns - L + 1], W, 1);
            for w = 1:W
                k = k + 1;
                seg = rx(starts(w) : starts(w) + L - 1);
                iq(1, :, k) = real(seg);
                iq(2, :, k) = imag(seg);

                lab.fd_hz(k)        = fd;
                lab.snr_db(k)       = snr;
                lab.phase_rad(k)    = phi;
                lab.start_sample(k) = starts(w) - 1;
                lab.frame_index(k)  = f - 1;
                lab.split(k)        = split(f);
                lab.ncellid(k)      = rmc.NCellID;
                lab.nframe(k)       = rmc.NFrame;
                lab.modulation(k)   = m - 1;
                lab.ocng(k)         = useOcng;
            end
        end

        w0 = (f0 - 1) * W + 1;
        h5write(out, '/iq', iq, [1 1 w0], [2 L nw]);
        names = fieldnames(lab);
        for i = 1:numel(names)
            h5write(out, ['/' names{i}], lab.(names{i}), w0, nw);
        end

        el = toc(t0);
        fprintf('frames %6d / %d  (%.0f s elapsed, ~%.0f s left)\n', ...
                f1, F, el, el * (F - f1) / f1);
    end

    write_attributes(out, cfg, fs, Ns);
    fprintf('Wrote %d windows to %s\n', N, out);
end

function split = assign_splits(F, fractions)
    fractions = fractions / sum(fractions);
    nTrain = round(fractions(1) * F);
    nVal   = round(fractions(2) * F);
    split  = zeros(F, 1, 'uint8');
    order  = randperm(F);
    split(order(nTrain+1 : nTrain+nVal)) = 1;
    split(order(nTrain+nVal+1 : end))    = 2;
end

function create_file(out, L, N)
    outDir = fileparts(out);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end
    if isfile(out)
        delete(out);
    end
    chunk = min(N, 1024);
    h5create(out, '/iq', [2 L N], 'Datatype', 'single', 'ChunkSize', [2 L chunk]);
    spec = {'fd_hz','single'; 'snr_db','single'; 'phase_rad','single'; ...
            'start_sample','int32'; 'frame_index','int32'; 'split','uint8'; ...
            'ncellid','int16'; 'nframe','int16'; 'modulation','uint8'; 'ocng','uint8'};
    for i = 1:size(spec, 1)
        h5create(out, ['/' spec{i,1}], N, 'Datatype', spec{i,2}, 'ChunkSize', chunk);
    end
end

function write_attributes(out, cfg, fs, Ns)
    h5writeatt(out, '/', 'sample_rate_hz',     fs);
    h5writeatt(out, '/', 'samples_per_frame',  int32(Ns));
    h5writeatt(out, '/', 'window_length',      int32(cfg.windowLength));
    h5writeatt(out, '/', 'windows_per_frame',  int32(cfg.windowsPerFrame));
    h5writeatt(out, '/', 'n_frames',           int32(cfg.nFrames));
    h5writeatt(out, '/', 'carrier_hz',         cfg.fc);
    h5writeatt(out, '/', 'fd_range_hz',        cfg.fdRange);
    h5writeatt(out, '/', 'snr_range_db',       cfg.snrRangeDb);
    h5writeatt(out, '/', 'rmc',                cfg.rc);
    h5writeatt(out, '/', 'modulations',        strjoin(cfg.modulations, ','));
    h5writeatt(out, '/', 'modulation_probs',   cfg.modulationProbs);
    h5writeatt(out, '/', 'ocng_prob',          cfg.ocngProb);
    h5writeatt(out, '/', 'seed',               int32(cfg.seed));
    h5writeatt(out, '/', 'matlab_version',     version);
end

function s = onoff(tf)
    if tf
        s = 'On';
    else
        s = 'Off';
    end
end
