function [aligned_ids, aligned_mzs, mzs] = alignEICs(mz_arrs, is_sample)
    % Identifies and aligns m/z peaks of different samples based on an input
    % tolerance (Config.MS_ACCURACY).
    initialize_config;

    center_mzs = chooseInitialCenters(mz_arrs, Config.MS_ACCURACY);
    [~, aligned_mzs] = Align_given_mz_centers(mz_arrs, center_mzs, Config.MS_ACCURACY);
    center_mzs = recalculateCenters(aligned_mzs);
    [aligned_ids, aligned_mzs] = Align_given_mz_centers(mz_arrs, center_mzs, Config.MS_ACCURACY);

    fprintf("MZ alignment log:\n");
    fprintf("\tBefore iterating, mzs: %d\n", numel(center_mzs));
    NUM_CORRECTION_ITERATIONS = 5;
    % k-means-like iteration process, except we also allow merging centers
    for i = 1:NUM_CORRECTION_ITERATIONS
        [aligned_mzs, aligned_ids] = rebalanceMzs(aligned_mzs, is_sample, aligned_ids);
    end

    mzs = recalculateCenters(aligned_mzs);
end

function [center_mzs] = chooseInitialCenters(mz_arrs, TOL)
    % picks initial center mzs according to all the mzs in mz_arrs.

    sample_num = length(mz_arrs);
    center_mzs = [];

    % prepare a pointer to the end of every mz_arr - they will later move
    % to the left with the algorithm.
    curr_idx = zeros(sample_num, 1);

    for i = 1:sample_num
        curr_idx(i) = numel(mz_arrs{i});
    end

    % gather unique mz by one linear sweep from right to left on mz_arrs
    while numel(curr_idx) > 0
        % locate maximum of all unchecked mzs
        for i = 1:numel(curr_idx)
            mz_arr = mz_arrs{i};
            max_vals(i) = mz_arr(curr_idx(i));
        end

        max_mz = max(max_vals);

        % for every sample, throw out those mzs which are now in TOL distance
        % from new max_mz added (should be MS_ACCURACY probably
        thrown_sum = 0;
        thrown_total = 0;

        for i = 1:numel(curr_idx)
            mz_arr = mz_arrs{i};
            to_throw = 0;

            % throw the next mz until we reach one which is out of TOL
            % range, or reach the start of the mz_arr
            while (to_throw < curr_idx(i)) && (max_mz - mz_arr(curr_idx(i) - to_throw)) <= 2 * TOL * max_mz
                thrown_sum = thrown_sum + mz_arr(curr_idx(i) - to_throw);
                thrown_total = thrown_total + 1;
                to_throw = to_throw + 1;
            end

            % here is a slight assumption that to_throw < 2, that is: a
            % situation where two mzs of the same sample are thrown, never
            % happens (risky, but empirically this never happens I think).
            % With this, the logic above could be simplified into an if
            % instead of while.

            curr_idx(i) = curr_idx(i) - to_throw;
        end

        % add the mean of all thrown mzs into center list
        center_mzs = [thrown_sum / thrown_total; center_mzs];

        % throw out "dead samples" - those which we finished sweeping
        mz_arrs = mz_arrs(curr_idx > 0);
        curr_idx = curr_idx(curr_idx > 0);
        max_vals = zeros(size(curr_idx));
    end

end

function [aligned_ids, aligned_mzs] = Align_given_mz_centers(mz_arrs, center_mzs, TOL)
    sample_num = length(mz_arrs);

    aligned_mzs = nan(sample_num, length(center_mzs));
    aligned_ids = nan(sample_num, length(center_mzs));

    curr_idx = zeros(sample_num, 1);

    for i = 1:sample_num
        curr_idx(i) = numel(mz_arrs{i});
    end

    live_samples = 1:sample_num;

    % for each center, find closest mz of each sample, if within TOL

    for center_idx = numel(center_mzs):-1:1
        curr_center = center_mzs(center_idx);

        for i = 1:numel(mz_arrs)
            closest_mz = Inf;
            mz_arr = mz_arrs{i};
            % we move the "sliding window" left in one of two scenarios:
            % 1. current mz is *not yet* in the TOL range of curr_center:
            %       center -------] TOL ........ *mz*
            % 2. current mz is in the TOL range, and is closer than the
            % previously chosen closest mz:
            %    TOL [-------*mz*--center--------------closest----] TOL

            while (curr_idx(i) > 0)
                curr_mz = mz_arr(curr_idx(i));

                if curr_mz > curr_center * (1 + TOL)
                    % -------curr_center------]TOL----curr_mz
                    curr_idx(i) = curr_idx(i) - 1;
                    continue;
                end

                % the threshold for which we declare a match
                dist_threshold = TOL * curr_center;

                if center_idx > 1
                    % make sure that mz is not already closer to next
                    % center
                    dist_threshold = min(dist_threshold, abs(curr_mz - center_mzs(center_idx - 1)));
                end

                % among all mzs inside dist_threshold, find the closest to
                % curr_center
                while abs(curr_mz - curr_center) < min(dist_threshold, abs(closest_mz - curr_center))
                    closest_mz = curr_mz;
                    curr_idx(i) = curr_idx(i) - 1;

                    if (curr_idx(i) > 0)
                        curr_mz = mz_arr(curr_idx(i));
                    end

                end

                break;
            end

            % when out of this loop, closest_mz is either the closest one
            % to the center, or Inf (if none were inside TOL range).
            if closest_mz ~= Inf
                aligned_mzs(live_samples(i), center_idx) = closest_mz;
                aligned_ids(live_samples(i), center_idx) = curr_idx(i) + 1;
            end

        end

        % throw out "dead samples" - those which we finished sweeping
        mz_arrs = mz_arrs(curr_idx > 0);
        live_samples = live_samples(curr_idx > 0);
        curr_idx = curr_idx(curr_idx > 0);
    end

end

function [center_mzs] = recalculateCenters(aligned_mzs)
    % this function is currently dumb, but it is here in case we want to
    % make it more complex someday.
    center_mzs = nanmean(aligned_mzs, 1)'; % make sure they are NaN and no 0.
    center_mzs = center_mzs(~isnan(center_mzs)); % in case we drop centers.
end
