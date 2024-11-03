function [grouping, pre_grouping] = groupPeaks(samples, aligned_ids, mzs)
    initialize_config;
    % The code below assumes that the scan rate of all sample is the same

    [max_num_scans, I] = max(cellfun(@(x) x.scan_num, samples));
    rt = samples{I}.sample_scans_rt;
    pre_grouping = struct;
    pre_grouping.mzs = mzs;
    pre_grouping.rt = rt;
    pre_grouping.selected_aligned_eics_arr = {};
    pre_grouping.summed_eics = {};
    pre_grouping.peaks_in_summed_eics = {};
    pre_grouping.baseline_intensity = [];

    % Identifies peaks using summed EICs and then performs peak grouping
    for m = 1:length(mzs)
        % Calculates summed EIC
        summed_eic = zeros(max_num_scans, 1);
        selected_aligned_ids = aligned_ids(:, m);

        for sample_id = 1:length(selected_aligned_ids)
            eic_id = selected_aligned_ids(sample_id);

            if isnan(eic_id)
                continue;
            end

            cur_eic = samples{sample_id}.eic_arr{eic_id};

            summed_eic(1:length(cur_eic)) = summed_eic(1:length(cur_eic)) + cur_eic;
        end

        % Finds peaks in the summed_eic
        [peak_list, ~, ~, ~, ~, ~, ~, baseline_intensity] = identifyPeaksInEIC(summed_eic, rt);

        pre_grouping.selected_aligned_eics_arr{end + 1} = selected_aligned_ids;
        pre_grouping.summed_eics{end + 1} = summed_eic;
        pre_grouping.peaks_in_summed_eics{end + 1} = peak_list;
        pre_grouping.baseline_intensity(end + 1) = baseline_intensity;
    end

    grouping = groupPeaksCore(samples, pre_grouping, Config.RT_TOL);
end
