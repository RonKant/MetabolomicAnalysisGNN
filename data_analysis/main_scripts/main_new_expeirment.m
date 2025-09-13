initialize_config;

mzxmlDirectory = "C:\Ron\research\data\new_experiment_6x6\Batch1\";
% mzxmlDirectory = "C:\Ron\research\data\Data_mzxml\20210728_1-9_drugs";

resultsPath = "C:\Ron\research\results\";
metadata = constructmetadata(mzxmlDirectory);
drugNames = unique(metadata(metadata.isDrug, :).drugName);

for drugName = drugNames'

    drugMetadata = metadata( ...
        (metadata.ionization == Config.ionization) ...
        & ((metadata.drugName == drugName) | (~metadata.isDrug)) ...
        , :);
    filePaths = cellfun(@(fileName) char(fullfile(mzxmlDirectory, fileName)), drugMetadata.fileName, 'UniformOutput', false);

    [LCMS, T, samples, grouping] = loadLCMS(filePaths, drugMetadata.isDrug);

    matricesDrug = LCMS.mat(:, metadata.isDrug);
    meanIntensity = mean(matricesDrug, 2);
    numNans = sum(matricesDrug == 0, 2);

    selectedIndices = (meanIntensity >= Config.HQ_MIN_MEAN_INTENSITY) & (numNans <= Config.HQ_MAX_NUM_NANS_SAMPLE);

    saveDir = fullfile(resultsPath, "stats_tables");
    mkdir(saveDir);
    savePath = fullfile(saveDir, drugName + "_statsTable.csv");
    writetable(T(selectedIndices, :), savePath);

end
