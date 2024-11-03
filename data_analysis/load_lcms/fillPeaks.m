function [LCMS, grouping_filled] = fillPeaks(LCMS, grouping, samples, is_sample)
    initialize_config;
    MIN_NUM_TO_FILL = floor(sum(is_sample) * Config.PEAK_FILLING_MIN_FRAC);
    eic_ids = grouping.aligned_eics_arr;

    grouping_filled = grouping;
    grouping_filled.selected_and_filled_peaks_arr = grouping_filled.selected_peaks_arr;
    %%
    for i = 1:length(LCMS.mz)
        rt = LCMS.rt(i);

        samples_data = LCMS.mat(i, find(is_sample));
        num_non_zero = sum(samples_data ~= 0);

        if num_non_zero < MIN_NUM_TO_FILL
            continue;
        end

        cur_aligned_eics = eic_ids{i};
        all_data = LCMS.mat(i, :);
        zero_ids = find(all_data == 0);

        curr_grouping_arr = grouping_filled.selected_and_filled_peaks_arr{i};

        for k = 1:length(zero_ids)
            j = zero_ids(k);

            if LCMS.mat(i, j) ~= 0
                continue;
            end

            sample = samples{j};
            loc = cur_aligned_eics(j);

            if ~isnan(loc)
                [~, loc_rt] = min(abs(sample.sample_scans_rt - rt));
                start_id = max(1, loc_rt - Config.PEAK_FILLING_NUM_SCANS);
                end_id = min(length(sample.sample_scans_rt), loc_rt + Config.PEAK_FILLING_NUM_SCANS);
                eic_in_rt = sample.eic_arr_smoothed{loc}(start_id:end_id);
                val = max(median(eic_in_rt), 0);

                LCMS.mat(i, j) = val;

                peak_area = trapz(sample.sample_scans_rt(start_id:end_id), eic_in_rt);
                LCMS.mat_peak_area(i, j) = peak_area;

                eic_in_rt = sort(eic_in_rt, 'descend');
                topk = eic_in_rt(1:min(3, numel(eic_in_rt)));
                LCMS.mat_top3_mean(i, j) = mean(topk);

                assert(~any(curr_grouping_arr(:, 1) == j)); % remove if slow

                new_row = [j, loc, nan, sample.sample_scans_rt(loc_rt), val, peak_area, start_id, end_id, LCMS.mat_top3_mean(i, j)];
                curr_grouping_arr = [curr_grouping_arr; new_row];
            end

        end

        [~, I] = sort(curr_grouping_arr(:, 1));
        curr_grouping_arr = curr_grouping_arr(I, :);

        grouping_filled.selected_and_filled_peaks_arr{i} = curr_grouping_arr;
    end

end
