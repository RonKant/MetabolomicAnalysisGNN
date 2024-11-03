function sample = extractEICs(sample, filter_eics)
    initialize_config;
    % Generates EIC (extracted ion chromatogram) for each m/z slice found by "Mz_slicing"
    scan_num = sample.scan_num;
    cluster_mz = sample.cluster_mz;
    sample_scans = sample.sample_scans;

    scan_index_arr = ones(scan_num, 1);
    eic_seq_num = zeros(length(cluster_mz), 1);
    eic_max_intensity = zeros(length(cluster_mz), 1);
    eic_arr = cell(length(cluster_mz), 1);
    eic_arr_mz = zeros(length(cluster_mz), 1);

    for x = 1:length(cluster_mz)
        min_mz = cluster_mz(x) - Config.MS_ACCURACY * cluster_mz(x);
        max_mz = cluster_mz(x) + Config.MS_ACCURACY * cluster_mz(x);
        eic = zeros(scan_num, 1);

        if x < length(cluster_mz)
            next_min = cluster_mz(x + 1) - Config.MS_ACCURACY * cluster_mz(x + 1);
        else
            next_min = Inf;
        end

        for y = 1:scan_num
            counter = scan_index_arr(y);

            while (counter < length(sample_scans{y}.mz))

                if (sample_scans{y}.mz(counter) < min_mz)
                    counter = counter + 1;
                    scan_index_arr(y) = scan_index_arr(y) + 1;
                    continue;
                end

                if (sample_scans{y}.mz(counter) < max_mz)
                    eic(y) = eic(y) + sample_scans{y}.intensity(counter);
                    %incresing scan index just in case there is no overlap with
                    %the next mz window
                    if sample_scans{y}.mz(counter) < next_min
                        scan_index_arr(y) = scan_index_arr(y) + 1;
                    end

                    counter = counter + 1;
                    continue;
                end

                break;
            end

        end

        v = find(eic == 0);

        if isempty(v)
            eic_seq_num(x) = length(eic);
        else
            v = [0; v; length(eic)];
            u = v(2:end) - v(1:end - 1);
            eic_seq_num(x) = max(u) - 1;
        end

        eic_arr_mz(x) = cluster_mz(x);
        eic_max_intensity(x) = max(eic);
        eic_arr{x} = eic;
    end

    if nargin < 3 || filter_eics == 1
        % choose EICs non-zero intensities in consecutive scans;
        % and maximal intensity greater than a predefiend threshold
        eic_used = find(eic_seq_num >= Config.EIC_CONSECUTIVE_SCANS & eic_max_intensity > Config.MIN_PEAK_INTENSITY);
    else
        eic_used = 1:length(eic_seq_num);
    end

    sample.eic_seq_num = eic_seq_num(eic_used);
    sample.eic_max_intensity = eic_max_intensity(eic_used);
    sample.eic_arr = eic_arr(eic_used);
    sample.eic_arr_mz = eic_arr_mz(eic_used);
