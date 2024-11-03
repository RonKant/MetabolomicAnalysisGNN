function [LCMS, grouping] = constructLCMSData(grouping, sample_labels)

    LCMS = struct;
    mz_str = strtrim(cellstr(num2str(grouping.mz')));
    rt_str = strtrim(cellstr(num2str(round(grouping.median_rt * 10) / 10)));

    feature_names = cell(length(mz_str), 1);

    for i = 1:length(feature_names)
        feature_names{i} = strcat('mz_', mz_str{i}, '_rt_', rt_str{i});
    end

    mat = zeros(length(mz_str), length(sample_labels));
    mat_peak_area = zeros(length(mz_str), length(sample_labels));
    mat_top3_mean = zeros(size(mat));

    for i = 1:length(grouping.mz)
        sample_ids = grouping.selected_peaks_arr{i}(:, 1);
        mat(i, sample_ids) = grouping.selected_peaks_arr{i}(:, 5);
        mat_peak_area(i, sample_ids) = grouping.selected_peaks_arr{i}(:, 6); % peak_area_top;
        mat_top3_mean(i, sample_ids) = grouping.selected_peaks_arr{i}(:, 9);
    end

    [feature_names, I] = unique(feature_names);

    LCMS.mz = grouping.mz(I)';
    LCMS.rt = grouping.median_rt(I);
    LCMS.feature_names = feature_names;
    LCMS.sample_labels = sample_labels;
    LCMS.mat = mat(I, :);
    LCMS.mat_peak_area = mat_peak_area(I, :);
    LCMS.baseline_fold_change = grouping.baseline_fold_change(I);
    LCMS.mat_top3_mean = mat_top3_mean(I, :);

    grouping_fields = fieldnames(grouping)';

    for i = 1:length(grouping_fields)
        field = grouping_fields{i};
        cur = grouping.(field);
        grouping.(field) = cur(I);
    end

end
