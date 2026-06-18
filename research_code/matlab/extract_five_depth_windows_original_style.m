function [features, windows, anchorPixel] = extract_five_depth_windows_original_style(CorrectedOCT, site, opts)
%EXTRACT_FIVE_DEPTH_WINDOWS_ORIGINAL_STYLE Five OCT depth-window features.
%
% This organized version supports both the manuscript anchor rule and manual
% inspection:
%   - automatic mode: skip the first 10 pixels, then search for the first local
%     peak inside a site-specific DEJ range;
%   - manual mode: supply opts.manualAnchorPixel to override automatic search.
%
% The output features are temporal curves: each column is the average OCT
% intensity inside one depth window over time.

arguments
    CorrectedOCT double
    site string = "wrist"
    opts.manualAnchorPixel double = NaN
    opts.offsets double = [8, 18, 28, 38, 48]
    opts.halfWidth double = 5
    opts.searchRange double = []
    opts.smoothWindow double = 5
end

Data = CorrectedOCT;
[nRows, nCols] = size(Data);
if nRows > nCols
    nTime = nRows;
    nDepth = nCols;
else
    Data = Data';
    [nTime, nDepth] = size(Data);
end

if ~isnan(opts.manualAnchorPixel)
    anchorPixel = round(opts.manualAnchorPixel);
else
    if isempty(opts.searchRange)
        switch lower(site)
            case {"arm", "wrist", "forearm"}
                searchRange = [27, 45];
            case "finger"
                searchRange = [55, 75];
            otherwise
                error('Unknown site. Use arm, wrist, forearm, finger, or provide opts.searchRange.');
        end
    else
        searchRange = opts.searchRange;
    end
    anchorPixel = detect_first_peak_after_cutoff(Data, searchRange, opts.smoothWindow);
end

centers = anchorPixel + opts.offsets(:);
windows = zeros(numel(centers), 2);
features = zeros(nTime, numel(centers));

for k = 1:numel(centers)
    startPixel = max(1, centers(k) - opts.halfWidth);
    stopPixel = min(nDepth, centers(k) + opts.halfWidth);
    windows(k, :) = [startPixel, stopPixel];
    features(:, k) = mean(Data(:, startPixel:stopPixel), 2, 'omitnan');
end
end

function anchorPixel = detect_first_peak_after_cutoff(Data, searchRange, smoothWindow)
depthProfile = mean(Data, 1, 'omitnan');
depthProfile = smoothdata(depthProfile, 'movmean', smoothWindow);

startPixel = max(10, round(searchRange(1)));
stopPixel = min(size(Data, 2) - 1, round(searchRange(2)));
if startPixel >= stopPixel
    error('Invalid first-peak search range.');
end

anchorPixel = NaN;
for z = startPixel:stopPixel
    if depthProfile(z - 1) < depthProfile(z) && depthProfile(z) >= depthProfile(z + 1)
        anchorPixel = z;
        break;
    end
end

if isnan(anchorPixel)
    [~, idx] = max(depthProfile(startPixel:stopPixel));
    anchorPixel = startPixel + idx - 1;
end
end
