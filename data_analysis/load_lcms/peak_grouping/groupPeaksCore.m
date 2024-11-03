function grouping = groupPeaksCore(samples, pre_grouping, RT_TOL)
    initialize_config;
    % Peak grouping
    % Asigns peaks in individual EICs to the peaks identified in the summed
    % EIC
    grouping = struct;
    grouping.mz = [];
    grouping.rt = [];
    grouping.summed_eic_id = [];
    grouping.peak_id_in_eic = [];
    % each row represents sample_id,eic_id, peak_id, rt ,intensity,peakAreaTop,
    % baseline_fold_change, peak_start_id,peak_end_id
    grouping.selected_peaks_arr = {};
    grouping.aligned_eics_arr = {};
    grouping.baseline_intensity = [];

    mzs = pre_grouping.mzs;

    for m = 1:length(pre_grouping.mzs)
        selected_aligned_ids = pre_grouping.selected_aligned_eics_arr{m};
        peak_list = pre_grouping.peaks_in_summed_eics{m};

        if numel(peak_list) > 0

            for p = 1:size(peak_list, 1)
                cur_rt = peak_list(p, 1);
                selected_peaks_arr = [];

                for sample_id = 1:length(selected_aligned_ids)
                    eic_id = selected_aligned_ids(sample_id);

                    if isnan(eic_id)
                        continue;
                    end

                    cur_peaks_rt = samples{sample_id}.peaks{eic_id}.rt;

                    % finds matched peak based on rt
                    [min_rt_diff, closest_peak_id] = min(abs(cur_rt - cur_peaks_rt));

                    if min_rt_diff < RT_TOL
                        selected_peaks_arr(end + 1, :) = [sample_id, eic_id, closest_peak_id, cur_peaks_rt(closest_peak_id), ...
                                                              samples{sample_id}.peaks{eic_id}.intensity(closest_peak_id), ...
                                                              samples{sample_id}.peaks{eic_id}.peak_area_top(closest_peak_id), ...
                                                              samples{sample_id}.peaks{eic_id}.peak_start_id(closest_peak_id), ...
                                                              samples{sample_id}.peaks{eic_id}.peak_end_id(closest_peak_id), ...
                                                              samples{sample_id}.peaks{eic_id}.peak_top3_mean(closest_peak_id)];
                    end

                end

                if ~isempty(selected_peaks_arr)
                    grouping.summed_eic_id(end + 1) = m;
                    grouping.peak_id_in_eic(end + 1) = p;
                    grouping.mz(end + 1) = mzs(m);
                    grouping.rt(end + 1) = cur_rt;
                    grouping.selected_peaks_arr{end + 1} = selected_peaks_arr;
                    grouping.aligned_eics_arr{end + 1} = selected_aligned_ids;
                    grouping.baseline_intensity(end + 1) = pre_grouping.baseline_intensity(m);
                end

            end

        end

    end

    grouping.median_rt = zeros(length(grouping.mz), 1);
    grouping.rt_MAD = zeros(length(grouping.mz), 1);
    grouping.median_intensity = zeros(length(grouping.mz), 1);
    grouping.group_size = zeros(length(grouping.mz), 1);
    grouping.baseline_fold_change = zeros(length(grouping.mz), 1);

    for i = 1:length(grouping.mz)
        grouping.median_rt(i) = median(grouping.selected_peaks_arr{i}(:, 4));
        grouping.rt_MAD(i) = mean(abs(grouping.median_rt(i) - grouping.selected_peaks_arr{i}(:, 4)));
        grouping.median_intensity(i) = median(grouping.selected_peaks_arr{i}(:, 5));
        grouping.group_size(i) = size(grouping.selected_peaks_arr{i}, 1);

        %median baseline fold change
        grouping.baseline_fold_change(i) = median(grouping.selected_peaks_arr{i}(:, 7));
    end

    % Remove repeated groups (same mz and a very close RT)
    selected = ones(length(grouping.mz), 1);

    for i = 2:length(grouping.mz)

        if grouping.summed_eic_id(i) == grouping.summed_eic_id(i - 1)

            if abs(grouping.median_rt(i) - grouping.median_rt(i - 1)) < Config.RT_UNIQUETOL
                % Selects the bigger group
                if grouping.group_size(i) < grouping.group_size(i - 1)
                    selected(i) = 0;
                elseif grouping.group_size(i) > grouping.group_size(i - 1)
                    selected(i - 1) = 0;
                else
                    % same group size - select by better intensity
                    [~, min_intensity_idx] = min([grouping.median_intensity([i - 1, i])]);
                    id_to_deselect = [i - 1, i];
                    id_to_deselect = id_to_deselect(min_intensity_idx);
                    selected(id_to_deselect) = 0; % deselect the lower intensity idx
                end

            end

        end

    end

    I = find(selected);
    grouping_fields = fieldnames(grouping)';

    for i = 1:length(grouping_fields)
        field = grouping_fields{i};
        cur = grouping.(field);
        grouping.(field) = cur(I);
    end

end
