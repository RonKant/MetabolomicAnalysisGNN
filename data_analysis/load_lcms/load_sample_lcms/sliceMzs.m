function sample = sliceMzs(sample)
    initialize_config;
    % Identify m/z slices
    % count number of ions in each m/z bin size of 0.001 / 10 Da
    scan_num = length(sample.scan);
    cluster_mz = [0:Config.MZ_SLICING_WIDTH:Config.MAX_MZ];
    cluster_ions = zeros(length(cluster_mz), 1);
    cluster_ind = [1:length(cluster_mz)];
    sample_scans_rt = zeros(scan_num, 1);
    sample_scans = cell(0);

    for i = 1:scan_num
        mz = sample.scan(i).peaks.mz(1:2:end);
        intensity = sample.scan(i).peaks.mz(2:2:end);

        sample_scans{i}.mz = mz;
        sample_scans{i}.intensity = intensity;
        sample_scans_rt(i) = str2double(sample.scan(i).retentionTime(3:end - 1));

        v = round(mz / Config.MZ_SLICING_WIDTH) + 1;

        for x = 1:length(v)
            cluster_ions(v(x)) = cluster_ions(v(x)) + 1;
        end

    end

    % define clusters m/z bins with a local maxima of ions(from one side at
    % least)
    cluster_center = ((cluster_ions(2:end - 1) >= cluster_ions(1:end - 2) & cluster_ions(2:end - 1) > cluster_ions(3:end)) | ...
        (cluster_ions(2:end - 1) > cluster_ions(1:end - 2) & cluster_ions(2:end - 1) >= cluster_ions(3:end)));
    cluster_center = [0; cluster_center; 0];

    t = find(cluster_center);
    cluster_mz = cluster_mz(t);
    cluster_ions = cluster_ions(t);
    cluster_ind = cluster_ind(t);

    cluster_ions_combined = cluster_ions;
    % remove overlapping clusters
    for y = 1:5

        for x = 1:length(cluster_mz) - 1

            if (cluster_mz(x) == 0)
                continue;
            end

            if checkMzDiff(cluster_mz(x + 1), cluster_mz(x), Config.MS_ACCURACY)

                if (cluster_ions(x) > cluster_ions(x + 1))
                    cluster_mz(x + 1) = 0;
                    cluster_ions_combined(x) = cluster_ions_combined(x) + cluster_ions_combined(x + 1);
                else
                    cluster_mz(x) = 0;
                    cluster_ions_combined(x + 1) = cluster_ions_combined(x + 1) + cluster_ions_combined(x);
                end

            end

        end

        t = find(cluster_mz ~= 0);
        cluster_mz = cluster_mz(t);
        cluster_ions = cluster_ions(t);
        cluster_ions_combined = cluster_ions_combined(t);
        cluster_ind = cluster_ind(t);
    end

    cluster_ions = cluster_ions_combined;
    sample.scan_num = scan_num;
    sample.sample_scans = sample_scans;
    sample.sample_scans_rt = sample_scans_rt;

    % Removes potential clusters with low number of ions
    selected_clustres = find(cluster_ions >= Config.EIC_CONSECUTIVE_SCANS);
    sample.cluster_mz = cluster_mz(selected_clustres);
    sample.cluster_ions = cluster_ions(selected_clustres);
    sample.cluster_ind = cluster_ind(selected_clustres);
