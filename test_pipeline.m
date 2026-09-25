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

%% 4. End-to-end file: shapes, labels, splits, reproducibility
cfg = pipeline_config();
cfg.nFrames = 40;
cfg.windowsPerFrame = 5;
cfg.framesPerChunk = 16;               % forces several chunked writes
cfg.fdRange = [3000 3000];             % fixed, inside the CP estimator's range
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
assert(all(fd == 3000), 'fd labels wrong');
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
fprintf('\nAll tests passed.\n');
