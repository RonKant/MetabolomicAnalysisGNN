function sample = findPeaks(sample)
    rt = sample.sample_scans_rt;

    sample.peaks = {};
    sample.eic_arr_smoothed = cell(size(sample.eic_arr));

    for i = 1:length(sample.eic_arr)
        eic = sample.eic_arr{i};
        [peak_list, eic_smooth, start_ids, end_ids, ~, ~, baseline_fold_change] = identifyPeaksInEIC(eic, rt);
        sample.eic_arr_smoothed{i} = eic_smooth;

        sample.peaks{i} = struct;
        sample.peaks{i}.eic_id = [];
        sample.peaks{i}.mz = [];
        sample.peaks{i}.rt = [];
        sample.peaks{i}.intensity = [];
        sample.peaks{i}.peak_area_top = [];
        sample.peaks{i}.baseline_fold_change = baseline_fold_change;
        sample.peaks{i}.peak_start_id = [];
        sample.peaks{i}.peak_end_id = [];

        sample.peaks{i}.peak_top3_mean = [];

        for p = 1:numel(peak_list) / 2
            sample.peaks{i}.eic_id(end + 1) = i;
            sample.peaks{i}.mz(end + 1) = sample.eic_arr_mz(i);
            sample.peaks{i}.rt(end + 1) = peak_list(p, 1);
            sample.peaks{i}.intensity(end + 1) = peak_list(p, 2);

            start_id = start_ids(p);
            end_id = end_ids(p);

            sample.peaks{i}.peak_start_id(end + 1) = start_id;
            sample.peaks{i}.peak_end_id(end + 1) = end_id;

            if (start_id == end_id)
                [~, peak_rt_id] = min(abs(peak_list(p, 1) - rt));
                start_id = peak_rt_id - 1;
                end_id = peak_rt_id + 1;
            end

            peak_area = trapz(rt(start_id:end_id), eic_smooth(start_id:end_id));
            sample.peaks{i}.peak_area_top(end + 1) = peak_area;

            eic = sample.eic_arr{i};
            eic_peak = eic(start_id:end_id);
            eic_peak = sort(eic_peak, 'descend');

            topk = eic_peak(1:min(3, numel(eic_peak)));

            sample.peaks{i}.peak_top3_mean(end + 1) = mean(topk);
        end

    end
