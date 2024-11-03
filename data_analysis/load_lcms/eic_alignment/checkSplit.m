function [med_dist_tols, idx1, idx2, C2] = checkSplit(x1, x2)
    % returns distance of medians in case of splitting, in units of tol
    % idx1 and idx2 is to which cluster each entry of x1, x2 goes in case of
    % split (respectively).

    is_x2 = exist('x2', 'var');

    if ~is_x2
        x2 = [];
    end

    % to which cluster each element goes
    idx1 = zeros(size(x1));
    idx2 = zeros(size(x2));

    not_nan_x1_idx = find(~isnan(x1));
    not_nan_x2_idx = ~isnan(x2);
    x = [x1; x2];
    x = x(~isnan(x));

    overlap = ~isnan(x1) & ~isnan(x2);

    % each separator must separate overlappers
    if any(overlap)
        separators = linspace(max(x1(overlap)), min(x2(overlap)), 10);
    else
        % can also reach here with non overlapping ones
        separators = linspace(min(x), max(x), 10);
    end

    separators = separators(2:end - 1); % do not include the closest overlap itself

    best_sep_id = 1;
    best_sep_score = -Inf;

    for s_id = 1:numel(separators)
        sep = separators(s_id);
        cluster1 = x(x <= sep);
        cluster2 = x(x > sep);

        sep_score = abs(mean(cluster2) - mean(cluster1));

        if sep_score > best_sep_score
            best_sep_score = sep_score;
            best_sep_id = s_id;
        end

    end

    best_separator = separators(best_sep_id);

    idx = (x > best_separator) + 1;
    C2 = [mean(x(idx == 1)); mean(x(idx == 2))];

    medoid_dist = abs(C2(1) - C2(2));
    tol = median(x) * 5/10 ^ 6;

    med_dist_tols = medoid_dist / tol;

    cluster_idx_for_x1 = idx(1:numel(not_nan_x1_idx));
    cluster_idx_for_x2 = idx(numel(not_nan_x1_idx) + 1:end);

    idx1(not_nan_x1_idx) = cluster_idx_for_x1;
    idx2(not_nan_x2_idx) = cluster_idx_for_x2;

end
