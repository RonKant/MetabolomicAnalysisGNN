function grouping = applyPeakAlignmentToGrouping(grouping, p_arr, initialize)

    if nargin < 3
        initialize = 0;
    end

    if initialize == 1
        grouping.rt_aligned_arr = cell(length(grouping.mz), 1);
        grouping.median_rt_aligned = zeros(length(grouping.mz), 1);

        for i = 1:length(grouping.mz)
            grouping.rt_aligned_arr{i} = grouping.selected_peaks_arr{i}(:, 4);
        end

    end

    for i = 1:length(grouping.mz)
        sample_ids_arr = grouping.selected_peaks_arr{i}(:, 1);
        sample_ids_unique = unique(grouping.selected_peaks_arr{i}(:, 1));

        for j = 1:length(sample_ids_unique)
            cur_id = sample_ids_unique(j);
            locs = sample_ids_arr == cur_id;
            grouping.rt_aligned_arr{i}(locs) = polyval(p_arr{cur_id}, grouping.rt_aligned_arr{i}(locs));
        end

        grouping.median_rt_aligned(i) = median(grouping.rt_aligned_arr{i});
    end

end
