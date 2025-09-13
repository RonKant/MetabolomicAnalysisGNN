function [LCMS, T, samples, grouping] = loadLCMS(filePaths, isDrug)
    initialize_config;
    samples = cell(length(filePaths), 1);

    % Loads and preprocesses each sample (EIC extraction, peak identification)
    tic;

    for i = 1:length(filePaths)
        fprintf('Loading sample %d / %d', i, length(filePaths));
        samples{i} = loadsampleLCMS(filePaths{i});
    end

    toc;

    fprintf('\nAligns mzs\n');

    % Aligns mz of detected EICs across samples
    tic;
    eicArray = cell(length(samples), 1);

    for i = 1:length(samples)
        eicArray{i} = samples{i}.eic_arr_mz;
    end

    [aligned_ids, ~, mzs] = alignEICs(eicArray, isDrug);

    toc;

    % Group peaks across samples (calculates summed EICs for a given m/z across
    % samples, performs peak detection in the summed EIC and then individual
    % EICs are assigned based on RT)
    fprintf('Peak Grouping\n');
    [grouping, pre_grouping] = groupPeaks(samples, aligned_ids, mzs);

    % Peak alignment using a global polynomial fitting
    fprintf('Calculates peak Alignment\n');
    [~, pre_grouping, poly_arr] = alignPeaks(samples, grouping, pre_grouping);

    % Regroups after alignment
    fprintf('Applies peak alignment\n');
    samples = applyPeakAlignmentToSample(samples, poly_arr);
    grouping = groupPeaksCore(samples, pre_grouping, Config.RT_TOL_AFTER_ALIGNMENT);

    % Constructs a struct with all feauters
    sample_labels = cell(length(filePaths), 1);

    for i = 1:length(filePaths)
        splited = strsplit(filePaths{i}, '\');
        name = strsplit(splited{end}, '.');
        name = name{1};
        sample_labels{i} = name;
    end

    fprintf('Constrcuts matrix\n');
    [LCMS, grouping] = constructLCMSData(grouping, sample_labels);

    fprintf('Applies peak filling\n');
    LCMS = fillPeaks(LCMS, grouping, samples, isDrug);

    T = LCMSToTable(LCMS);
end

function T = LCMSToTable(LCMS)

    T = array2table(LCMS.mat, 'VariableNames', LCMS.sample_labels);
    T_mz = cell2table(LCMS.feature_names, 'VariableNames', {'metabolite'});
    T_mz2 = array2table([LCMS.mz, LCMS.rt], 'VariableNames', {'mz', 'rt'});
    T_baseline = array2table([LCMS.baseline_fold_change], 'VariableNames', {'baseline_fold_change'});

    T = [T_mz T_mz2 T, T_baseline];
end
