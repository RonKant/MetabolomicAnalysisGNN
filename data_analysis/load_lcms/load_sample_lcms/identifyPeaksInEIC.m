function [peak_list, eic_smooth, start_ids, end_ids, PFWHH, PExt, baseline_fold_change, baseline_intensity] = identifyPeaksInEIC(eic, rt)
    initialize_config;

    % baseline intensity estimation (remove top 20% scans with highest
    % intensity; gaussian smooth, and median intensity
    baseline_prctile = prctile(eic, Config.BASELINE_DROP_TOP_INTENSITY_PER);
    eic_baseline = eic;
    eic_baseline(eic_baseline > baseline_prctile) = median(eic_baseline(eic_baseline <= baseline_prctile)); % % Added = to < due to 0 values
    eic_baseline_smooth = smoothdata(eic_baseline, 'gaussian', Config.BASELING_SMOOTH_WINDOW);
    baseline_intensity = median(eic_baseline_smooth); % this also takes values that are 0 TODO?

    % find local peaks in smoothed intensity curve that are
    % MIN_SIGNAL_BASELINE_RATIO fold higher than baseline intensity
    eic_smooth = max(sgolayfilt(eic, Config.EIC_SMOOTH_ORDER, Config.EIC_SMOOTH_FRAMELEN), 0);

    %ANOTHER PEAK DETECTION METHOD
    min_peak_intensity = max(Config.MIN_PEAK_INTENSITY, baseline_intensity * Config.MIN_SIGNAL_BASELINE_RATIO);
    [peak_list, PFWHH, PExt] = mspeaks(rt, eic_smooth, 'HeightFilter', min_peak_intensity, 'Denoising', false, ...
        'PeakLocation', Config.PF_PEAK_LOCATION_VALUE, ...
        'OverSegmentationFilter', Config.PF_OVERSEGMENTATION_VALUE);

    if size(peak_list, 2) == 0
        peak_list = [];
    end

    if size(peak_list, 1) > 2
        [peak_list, PFWHH, PExt] = merge_peak_fractions(peak_list, PFWHH, PExt, Config);
    end

    start_ids = zeros(size(peak_list, 1), 1);
    end_ids = zeros(size(peak_list, 1), 1);

    baseline_fold_change = [];

    if ~isempty(peak_list)
        baseline_fold_change = peak_list(:, 2) ./ baseline_intensity;

        %Filter peaks using a "local baseline"
        local_baseline = zeros(size(peak_list, 1), 1);
        location_in_local_sorting = zeros(size(local_baseline));

        for i = 1:size(peak_list, 1)
            [~, start_id] = min(abs(PExt(i, 1) - rt));
            [~, end_id] = min(abs(PExt(i, 2) - rt));

            peak_eic = eic_smooth(start_id:end_id);
            first_non_zero = find(peak_eic, 1, 'first');
            last_non_zero = find(peak_eic, 1, 'last');
            end_id = start_id + last_non_zero - 1;
            start_id = start_id + first_non_zero - 1;

            start_id_local_baseline = max(1, start_id - Config.LOCAL_BASELINE_SCANS);
            end_id_local_baseline = min(length(rt), end_id + Config.LOCAL_BASELINE_SCANS);
            local_baseline(i) = median(eic_smooth([start_id_local_baseline:start_id, end_id:end_id_local_baseline]));

            % check that the peak is maximal
            location_in_local_sorting(i) = 1 + sum(eic_smooth([start_id_local_baseline:end_id_local_baseline]) > ...
                peak_list(i, 2));

            start_ids(i) = start_id;
            end_ids(i) = end_id;
        end

        %Remove peaks from the first seconds/scans
        selected_ids = find(peak_list(:, 1) > Config.MIN_RT - Config.RT_TOL & ...
            peak_list(:, 2) > Config.MIN_SIGNAL_LOCAL_BASELINE_RATIO .* local_baseline & ...
            location_in_local_sorting == 1);

        peak_list = peak_list(selected_ids, :);
        PFWHH = PFWHH(selected_ids, :);
        PExt = PExt(selected_ids, :);
        baseline_fold_change = baseline_fold_change(selected_ids);
        start_ids = start_ids(selected_ids);
        end_ids = end_ids(selected_ids);
    end

end

function [new_peak_list, new_PFWHH, new_PExt] = merge_peak_fractions(peak_list, PFWHH, PExt, Config)
    new_peak_list = []; new_PFWHH = []; new_PExt = [];

    left_peak_idx = 1;

    while (left_peak_idx <= size(peak_list, 1))
        left_peak_rt = peak_list(left_peak_idx, 1);

        right_peak_idx = left_peak_idx + 1;

        while (right_peak_idx <= size(peak_list, 1)) ...
                && peak_list(right_peak_idx, 1) - left_peak_rt <= Config.PEAK_FINDING_MERGE_MAX_RT
            prev_mean_intensity = mean(peak_list(left_peak_idx:right_peak_idx - 1, 2));

            if abs(prev_mean_intensity - peak_list(right_peak_idx, 2)) > ...
                    Config.PEAK_FINDING_MERGE_INTENSITY_DIFF_PRCNT * 0.01 * prev_mean_intensity
                break
            end

            right_peak_idx = right_peak_idx + 1;
        end

        right_peak_idx = right_peak_idx - 1; % last one is the one who did not enter

        % here left_peak_idx:right_peak_idx are the ones to merge
        merged_peak_rt = mean(peak_list(left_peak_idx:right_peak_idx, 1));
        merged_peak_intensity = max(peak_list(left_peak_idx:right_peak_idx, 2));
        % it is max so the location_in_sorted criterion does not fail.
        % After new mspeaks you can change this

        new_peak_list = [new_peak_list; merged_peak_rt, merged_peak_intensity];
        new_PFWHH = [new_PFWHH; PFWHH(left_peak_idx, 1), PFWHH(right_peak_idx, 2)];
        new_PExt = [new_PExt; PExt(left_peak_idx, 1), PFWHH(right_peak_idx, 2)];

        left_peak_idx = right_peak_idx + 1;
    end

end
