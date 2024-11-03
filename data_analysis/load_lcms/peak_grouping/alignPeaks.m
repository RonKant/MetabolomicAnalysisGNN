function [grouping, pre_grouping, poly_arr] = alignPeaks(samples, grouping, pre_grouping)
    initialize_config;

    % Choose high quality groups for alignment
    groups_selected_num_peaks = find(grouping.group_size >= Config.MIN_PEAKS_IN_GROUP);

    % beep
    % Top intensities groups
    intensities_in_selected = grouping.median_intensity(groups_selected_num_peaks);
    [~, I] = sort(intensities_in_selected, 'desc');

    top_ids = I(1:Config.MAX_NUM_GROUPS);
    groups_selected = groups_selected_num_peaks(top_ids);

    % Initialize fields of aligned rt
    p_identity_arr = repmat({[1, 0]}, length(samples), 1);

    grouping = applyPeakAlignmentToGrouping(grouping, p_identity_arr, 1);

    poly_arr = cell(length(samples), 1);

    for i = 1:Config.MAX_ALIGN_ITER

        if i == 1
            % Arranges data for alignment
            [rt_detected, rt_target, sample_ids] = Arrange_data_for_alignment(grouping, groups_selected);
        end

        cur_poly_arr = p_identity_arr;

        for s = 1:length(samples)
            p = polyfit(rt_detected(sample_ids == s), rt_target(sample_ids == s), Config.POLY_DEGREE);
            poly_arr{s}{end + 1} = p;
            cur_poly_arr{s} = p;
        end

        grouping = applyPeakAlignmentToGrouping(grouping, cur_poly_arr);

        prev_rt_detected = rt_detected;
        [rt_detected, rt_target, sample_ids] = Arrange_data_for_alignment(grouping, groups_selected);

        if prctile(abs(rt_detected - prev_rt_detected), Config.ALIGN_CONVERGENCE_PRCTILE) < Config.ALIGN_CONVERGENCE_TRESH %Convergence
            break;
        end

    end

    % Update the pre_grouping with the median_rt of the aligned peaks for
    % regrouping after alignment
    % this can correct wrong rt detected in the "summed eic"; Another approach might
    % be to recalculate summed eic after alignment
    pre_grouping = Update_pre_grouping(grouping, pre_grouping);
end

function [rt_detected, rt_target, sample_ids] = Arrange_data_for_alignment(grouping, groups_selected)
    % for the selected groups extracts the measured rt of ecah peak and the
    % median rt of the group
    rt_detected = [];
    rt_target = []; % Regreession target
    sample_ids = [];

    for i = 1:length(groups_selected)
        group_id = groups_selected(i);
        cur_group_median_rt = grouping.median_rt_aligned(group_id);
        rt_target = [rt_target; cur_group_median_rt .* ones(grouping.group_size(group_id), 1)];
        rt_detected = [rt_detected; grouping.rt_aligned_arr{group_id}];
        sample_ids = [sample_ids; grouping.selected_peaks_arr{group_id}(:, 1)];
    end

end

function pre_grouping = Update_pre_grouping(grouping, pre_grouping)
    initialize_config;

    for i = 1:length(pre_grouping.peaks_in_summed_eics)
        pre_grouping.peaks_in_summed_eics{i}(:, 1) = NaN;
    end

    for i = 1:length(grouping.summed_eic_id)
        med_rt_aligned = grouping.median_rt_aligned(i);
        eic_id = grouping.summed_eic_id(i);
        peak_id = grouping.peak_id_in_eic(i);
        pre_grouping.peaks_in_summed_eics{eic_id}(peak_id, 1) = med_rt_aligned;
    end

    for i = 1:length(pre_grouping.peaks_in_summed_eics)
        rts = pre_grouping.peaks_in_summed_eics{i}(:, 1);
        ids = find(~isnan(rts));
        pre_grouping.peaks_in_summed_eics{i} = pre_grouping.peaks_in_summed_eics{i}(ids, :);
    end

end
