function [CorrectedOCT, AlignedOCT, info] = biopac_preprocess_original_style(OCTSignal, opts)
%BIOPAC_PREPROCESS_ORIGINAL_STYLE Organized version of the internal Bio-PAC code.
%
% Input
%   OCTSignal : depth-time OCT signal with rows as time points and columns as
%               depth pixels after surface flattening and lateral averaging.
%   opts      : optional struct with fields:
%               depthLimit, nSegments, maxSlack, nRefLayers,
%               refSigma, currentSigma.
%
% Output
%   CorrectedOCT : Bio-PAC-corrected OCT signal.
%   AlignedOCT   : morphology-aligned OCT signal before optical decoupling.
%   info         : intermediate variables for inspection.
%
% This code keeps the original algorithmic choices:
%   1. Crop to the first 200 depth pixels.
%   2. Use abs(hilbert(A-scan)) and Gaussian smoothing to form envelopes.
%   3. Estimate one shift per depth segment by whole-segment cross-correlation.
%   4. Interpolate segment shifts with PCHIP to form a continuous warping field.
%   5. Average the first nRefLayers as an epidermal fingerprint N_epi(t).
%   6. Z-score N_epi(t) into P(t), regress every depth curve on P(t), and
%      subtract only alpha_z * P(t).

arguments
    OCTSignal double
    opts.depthLimit (1,1) double = 200
    opts.nSegments (1,1) double = 10
    opts.maxSlack (1,1) double = 30
    opts.nRefLayers (1,1) double = 10
    opts.refSigma (1,1) double = 10
    opts.currentSigma (1,1) double = 5
end

if isempty(OCTSignal)
    error('OCTSignal is empty.');
end

depthLimit = min(opts.depthLimit, size(OCTSignal, 2));
OCTSignal = OCTSignal(:, 1:depthLimit);
[nSamples, nDepth] = size(OCTSignal);

targetLength = nDepth;
segmentEdges = round(linspace(1, targetLength, opts.nSegments + 1));
segmentCenters = (segmentEdges(1:end-1) + segmentEdges(2:end)) / 2;

refScan = OCTSignal(1, :);
refEnv = smoothdata(abs(hilbert(refScan)), 'gaussian', opts.refSigma);

AlignedOCT = zeros(nSamples, targetLength);
segmentShifts = zeros(nSamples, opts.nSegments);
warpingFields = zeros(nSamples, targetLength);

for i = 1:nSamples
    currentScan = OCTSignal(i, :);
    currentEnv = smoothdata(abs(hilbert(currentScan)), 'gaussian', opts.currentSigma);

    localShifts = zeros(1, opts.nSegments);
    for s = 1:opts.nSegments
        idx = segmentEdges(s):segmentEdges(s + 1);
        [corrValues, lags] = xcorr(refEnv(idx), currentEnv(idx), opts.maxSlack);
        [~, maxIdx] = max(corrValues);
        localShifts(s) = lags(maxIdx);
    end

    pixelShifts = interp1(segmentCenters, localShifts, 1:targetLength, 'pchip', 'extrap');
    queryPoints = (1:targetLength) + pixelShifts;
    AlignedOCT(i, :) = interp1(1:nDepth, currentScan, queryPoints, 'linear', 0);

    segmentShifts(i, :) = localShifts;
    warpingFields(i, :) = pixelShifts;
end

nRefLayers = min(opts.nRefLayers, targetLength);
epidermalRegion = AlignedOCT(:, 1:nRefLayers);
N_epi = mean(epidermalRegion, 2);

if std(N_epi) == 0
    P = zeros(size(N_epi));
else
    P = (N_epi - mean(N_epi)) / std(N_epi);
end

CorrectedOCT = zeros(size(AlignedOCT));
alpha = zeros(1, targetLength);
for z = 1:targetLength
    depthCurve = AlignedOCT(:, z);
    if std(P) == 0
        CorrectedOCT(:, z) = depthCurve;
        continue;
    end
    coeffs = polyfit(P, depthCurve, 1);
    alpha(z) = coeffs(1);
    CorrectedOCT(:, z) = depthCurve - alpha(z) * P;
end

info = struct();
info.segmentShifts = segmentShifts;
info.warpingFields = warpingFields;
info.epidermalFingerprintRaw = N_epi;
info.epidermalFingerprintZscore = P;
info.depthCouplingAlpha = alpha;
info.parameters = opts;
end
