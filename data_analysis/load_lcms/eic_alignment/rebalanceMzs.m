function [T_new, aligned_ids_new] = rebalanceMzs(T, sample_idx, aligned_ids)
    T_new = T;
    aligned_ids_new = aligned_ids;

    for i = 1:size(T, 2) - 1
        curr = T_new(:, i);

        if all(isnan(curr))
            continue % it is a result of rebalancing
        end

        next = T_new(:, i + 1);

        med_curr = median(curr(~isnan(curr)));
        med_next = median(next(~isnan(next)));

        if (abs(min(next) - max(curr)) / (med_curr * 5/10 ^ 6)) > 2
            % more than 2 TOL - not worth looking
            continue
        end

        no_overlap = all(isnan(curr) | isnan(next));

        if no_overlap && abs(max(next) - min(curr)) / (min(curr) * 5/10 ^ 6) < 3 % if more than 3 TOL - better split smart
            curr_next_mat = [curr, next]; % for max
            [T_new(:, i), I] = max(curr_next_mat, [], 2, 'linear');
            T_new(:, i + 1) = nan(size(T_new(:, i + 1)));
            curr_next_ids_mat = aligned_ids(:, [i, i + 1]);

            aligned_ids_new(:, i) = curr_next_ids_mat(I);
            aligned_ids_new(:, i + 1) = nan(size(aligned_ids_new(:, i + 1)));
        else
            % if no overlap, go straight to merge (skip this if)
            med_dist = abs(med_next - med_curr) / (med_curr * 5/10 ^ 6);

            if med_dist > 2
                % more strict closeness if we wish to rebalance columns
                continue
            end

            f_sample_idx = find(sample_idx);
            dual_coverage = sum(~isnan(T(f_sample_idx, i)) | ~isnan(T_new(f_sample_idx, i + 1))) / numel(f_sample_idx);
            % total samples containing values (samples only, no blanks)
            if dual_coverage < 0.5
                % not worth merging if even together this is a bad pair
                continue
            end

            [~, idx_curr, idx_next, medoids] = checkSplit(curr, next);

            if medoids(1) < medoids(2)
                min_medoid = 1;
                max_medoid = 2;
            else
                min_medoid = 2;
                max_medoid = 1;
            end

            if any((idx_curr == max_medoid & idx_next == min_medoid))
                error("in mz alignment rebalancing");
                % contradicts some of my assumptions
            end

            medoid_overlap = (idx_curr == idx_next) & (idx_next ~= 0); % can't have this situation
            idx_curr(medoid_overlap) = min_medoid;
            idx_next(medoid_overlap) = max_medoid;

            T_new(:, i:i + 1) = nan(size(T_new(:, i:i + 1))); % important
            aligned_ids_new(:, i:i + 1) = nan(size(aligned_ids_new(:, i:i + 1)));

            curr_ids = aligned_ids(:, i);
            next_ids = aligned_ids(:, i + 1);

            T_new(idx_curr == min_medoid, i) = curr(idx_curr == min_medoid);
            T_new(idx_next == min_medoid, i) = next(idx_next == min_medoid);
            aligned_ids_new(idx_curr == min_medoid, i) = curr_ids(idx_curr == min_medoid);
            aligned_ids_new(idx_next == min_medoid, i) = next_ids(idx_next == min_medoid);

            T_new(idx_curr == max_medoid, i + 1) = curr(idx_curr == max_medoid);
            T_new(idx_next == max_medoid, i + 1) = next(idx_next == max_medoid);
            aligned_ids_new(idx_curr == max_medoid, i + 1) = curr_ids(idx_curr == max_medoid);
            aligned_ids_new(idx_next == max_medoid, i + 1) = next_ids(idx_next == max_medoid);

            continue % important - out of the if is the no overlap case
        end

    end

    aligned_ids_new = aligned_ids_new(:, sum(~isnan(T_new)) > 0);
    T_new = T_new(:, sum(~isnan(T_new)) > 0); % remove nan columns
end
