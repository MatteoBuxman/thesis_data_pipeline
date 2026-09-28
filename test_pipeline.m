%TEST_PIPELINE Sanity checks for SYNTHESIZE_FRAME and GENERATE_DATASET.
%   Run from the project folder: matlab -batch test_pipeline

rng(1, 'twister');
rmc = lteRMCDL('R.4');
rmc.TotSubframes = 10;
rmc.OCNGPDSCHEnable = 'On';

%% 1. Normalisation: noiseless, zero-Doppler frame has unit power
[rx, tx, fs] = synthesize_frame(rmc, 0, 0, Inf);
assert(abs(mean(abs(tx).^2) - 1) < 1e-12, 'tx not unit power');
assert(max(abs(rx - tx)) < 1e-12, 'zero Doppler, zero phase, no noise should be a no-op');
assert(fs == 1.92e6 && numel(rx) == 19200, 'unexpected sample rate or frame length');
fprintf('PASS  unit power, fs = %.2f MHz, %d samples/frame\n', fs/1e6, numel(rx));

%% 2. Doppler sign and size: toolbox CP-based estimator recovers fd
for fd = [3000 -5000]
    rx  = synthesize_frame(rmc, fd, 2*pi*rand, Inf);
    est = lteFrequencyOffset(rmc, rx);
    assert(abs(est - fd) < 10, 'fd = %g Hz estimated as %g Hz', fd, est);
    fprintf('PASS  fd = %+6d Hz, lteFrequencyOffset = %+9.2f Hz\n', fd, est);
end

%% 3. SNR: measured noise power matches the requested SNR
snrDb = 10;
rng(7); [rx, tx] = synthesize_frame(rmc, 1234, 0.5, snrDb);
rng(7); clean    = synthesize_frame(rmc, 1234, 0.5, Inf);
noise = rx - clean;
measured = 10*log10(mean(abs(tx).^2) / mean(abs(noise).^2));
assert(abs(measured - snrDb) < 0.2, 'SNR %.2f dB, expected %g dB', measured, snrDb);
fprintf('PASS  requested SNR %g dB, measured %.2f dB\n', snrDb, measured);

%% 3b. Vector Doppler input: a constant vector matches the scalar path
rng(9); a = synthesize_frame(rmc, 2500, 0.3, Inf);
rng(9); b = synthesize_frame(rmc, 2500 * ones(19200, 1), 0.3, Inf);
assert(max(abs(a - b)) < 1e-9, 'vector fd path disagrees with scalar path');
fprintf('PASS  constant fd vector reproduces the scalar-fd frame\n');

%% 4. End-to-end file: shapes, labels, splits, reproducibility
cfg = pipeline_config();
cfg.nFrames = 40;
cfg.windowsPerFrame = 5;
cfg.framesPerChunk = 16;               % forces several chunked writes
cfg.labelSampling = 'uniform';
cfg.fdRange = [3000 3000];             % fixed, inside the CP estimator's range
cfg.trajectoryProbs = [1 0 0 0 0];     % constant Doppler only
cfg.ampScaleProb = 0;
cfg.snrRangeDb = [200 200];            % effectively noiseless
cfg.outFile = fullfile(tempdir, 'lte_doppler_test.h5');
generate_dataset(cfg);

iq = h5read(cfg.outFile, '/iq');
N  = cfg.nFrames * cfg.windowsPerFrame;
assert(isequal(size(iq), [2 cfg.windowLength N]), 'iq has wrong shape');
fd    = h5read(cfg.outFile, '/fd_hz');
start = double(h5read(cfg.outFile, '/start_sample'));
frame = h5read(cfg.outFile, '/frame_index');
split = h5read(cfg.outFile, '/split');
assert(all(abs(fd - 3000) < 1e-3), 'fd labels wrong');
epsL = h5read(cfg.outFile, '/eps');
qL   = h5read(cfg.outFile, '/q_int');
dL   = h5read(cfg.outFile, '/delta');
assert(all(qL == 0) && all(abs(epsL - 0.2) < 1e-6) && all(abs(dL - 0.2) < 1e-6), ...
       'eps / q / delta labels wrong for fd = 3000 Hz');
assert(all(start >= 0 & start <= 19200 - cfg.windowLength), 'start out of range');
for f = unique(frame).'
    assert(numel(unique(split(frame == f))) == 1, 'frame %d split across sets', f);
end
fprintf('PASS  file shape [%s], labels in range, splits per frame\n', num2str(size(iq)));

% Each window's CP pairs must rotate by exactly 2*pi*fd*Nfft/fs.
nfft = 128;
cpLen = [10 9 9 9 9 9 9];
symStart = cumsum([0, repmat(cpLen + nfft, 1, 1)]);
symStart = symStart(1:end-1);
slotLen = sum(cpLen + nfft);
cpIdx = [];                                % 0-based CP sample indices in the frame
for s = 0:19
    for l = 1:7
        cpIdx = [cpIdx, s*slotLen + symStart(l) + (0:cpLen(l)-1)]; %#ok<AGROW>
    end
end
errs = nan(N, 1);
winPower = zeros(N, 1);
for k = 1:N
    x = complex(double(iq(1,:,k)), double(iq(2,:,k))).';
    winPower(k) = mean(abs(x).^2);
    n = cpIdx(cpIdx >= start(k) & cpIdx + nfft <= start(k) + cfg.windowLength - 1) - start(k);
    if isempty(n)
        continue                           % no complete CP pair inside this window
    end
    rot = angle(sum(conj(x(n+1)) .* x(n+1+nfft)));
    errs(k) = rot * fs / (2*pi*nfft) - fd(k);
end
checked = ~isnan(errs);
assert(max(abs(errs(checked))) < 1, 'window Doppler disagrees with label (max err %.2f Hz)', max(abs(errs(checked))));
fprintf('PASS  %d/%d windows with a CP pair rotate at their fd label (max err %.2e Hz)\n', ...
        nnz(checked), N, max(abs(errs(checked))));

% With OCNG on, no window should be (near-)silent.
assert(min(winPower) > 0.1, 'silent window found (power %.2e)', min(winPower));
fprintf('PASS  no silent windows (min window power %.2f)\n', min(winPower));

iq2 = iq;
generate_dataset(cfg);
assert(isequal(iq2, h5read(cfg.outFile, '/iq')), 'same seed produced different data');
fprintf('PASS  same seed reproduces identical data\n');

delete(cfg.outFile);

%% 5. Label sampling: balanced integer classes, delta in [-0.5, 0.5)
cfg = pipeline_config();
rng(3);
K = 2*cfg.qMax + 1;
n = 7000 * K;
e = zeros(n, 1);
for i = 1:n
    e(i) = sample_doppler(cfg) / cfg.subcarrierSpacing;
end
q = floor(e + 0.5);
d = e - q;
counts = histcounts(q, (-cfg.qMax - 0.5):(cfg.qMax + 0.5));
assert(all(abs(counts / n - 1/K) < 0.01), 'integer classes not balanced: %s', mat2str(counts));
assert(all(d >= -0.5 & d < 0.5), 'delta outside [-0.5, 0.5)');
fprintf('PASS  %d balanced classes (counts %s), delta in [-0.5, 0.5)\n', K, mat2str(counts));

%% 6. Trajectory families behave as specified
Ns = 19200; fsT = 1.92e6;
clip = doppler_label_range(cfg);
for t = 1:numel(cfg.trajectoryTypes)
    type = cfg.trajectoryTypes{t};
    for trial = 1:200
        fd0 = sample_doppler(cfg);
        [fi, info] = doppler_trajectory(type, fd0, Ns, fsT, cfg);
        assert(numel(fi) == Ns && all(isfinite(fi)), '%s: bad output', type);
        assert(all(fi >= clip(1) & fi <= clip(2)), '%s: outside label range', type);
        jumps = find(diff(fi) ~= 0);
        switch type
            case 'constant'
                assert(isempty(jumps), 'constant trajectory varies');
            case {'linear', 'sinusoid'}
                % clipping can only reduce the slope
                assert(max(abs(diff(fi))) * fsT <= cfg.fdRateMax * (1 + 1e-6), ...
                       '%s: rate exceeds fdRateMax', type);
            case 'step'
                assert(numel(jumps) <= 1, 'step trajectory has %d jumps', numel(jumps));
                if ~isempty(jumps)
                    assert(jumps + 1 == info.stepIndex, 'stepIndex does not match the jump');
                end
            case 'randomwalk'
                assert(numel(unique(fi)) <= cfg.symbolsPerFrame, ...
                       'random walk changes within an OFDM symbol');
        end
    end
end
fprintf('PASS  trajectory families: range, rate, step and symbol structure\n');

%% 7. End-to-end with time-varying Doppler: window labels match the signal
cfg = pipeline_config();
cfg.nFrames = 60;
cfg.windowsPerFrame = 5;
cfg.framesPerChunk = 20;
cfg.trajectoryProbs = [0 1 1 1 1];     % every non-constant family
cfg.ampScaleProb = 0.5;
cfg.snrRangeDb = [200 200];
cfg.outFile = fullfile(tempdir, 'lte_doppler_test_traj.h5');
generate_dataset(cfg);

iq    = h5read(cfg.outFile, '/iq');
fd    = double(h5read(cfg.outFile, '/fd_hz'));
start = double(h5read(cfg.outFile, '/start_sample'));
traj  = h5read(cfg.outFile, '/trajectory');
stepW = h5read(cfg.outFile, '/step_in_window');
qL    = double(h5read(cfg.outFile, '/q_int'));
dL    = double(h5read(cfg.outFile, '/delta'));
epsL  = double(h5read(cfg.outFile, '/eps'));
assert(all(abs(qL) <= cfg.qMax) && all(dL >= -0.5 & dL < 0.5), ...
       'stored q / delta outside {-qMax..qMax} x [-0.5, 0.5)');
assert(all(qL + dL == epsL), 'stored q + delta ~= eps');
assert(all(abs(epsL * cfg.subcarrierSpacing - fd) < 0.02), 'eps inconsistent with fd_hz');
N = cfg.nFrames * cfg.windowsPerFrame;
tol = [1, 1, 1, 1, 6*cfg.rwSigmaHz];   % Hz, per type (constant ... randomwalk)
errs = nan(N, 1);
for k = 1:N
    if stepW(k)
        continue                           % label is a mix of two Dopplers
    end
    x = complex(double(iq(1,:,k)), double(iq(2,:,k))).';
    nn = cpIdx(cpIdx >= start(k) & cpIdx + nfft <= start(k) + cfg.windowLength - 1) - start(k);
    if isempty(nn)
        continue
    end
    % A CP pair only sees fd modulo fs/nfft = 15 kHz (one subcarrier), so
    % compare the wrapped phase: this checks delta; q holds by construction.
    rot = angle(sum(conj(x(nn+1)) .* x(nn+1+nfft)));
    phErr = angle(exp(1j * (rot - 2*pi*fd(k)*nfft/fs)));
    errs(k) = abs(phErr * fs / (2*pi*nfft)) / tol(traj(k) + 1);
end
checked = ~isnan(errs);
assert(max(errs(checked)) < 1, 'window label disagrees with signal (worst %.2f x tolerance)', ...
       max(errs(checked)));
fprintf('PASS  %d/%d windows: CP rotation matches the window-mean Doppler label\n', ...
        nnz(checked), N);
delete(cfg.outFile);

fprintf('\nAll tests passed.\n');
