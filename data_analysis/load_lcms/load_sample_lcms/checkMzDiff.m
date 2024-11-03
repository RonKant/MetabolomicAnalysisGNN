function same = checkMzDiff(mz1, mz2, MS_ACCURACY)
    same = 0;

    allowed_diff = MS_ACCURACY * max(mz1, mz2);

    if abs(mz1 - mz2) < allowed_diff
        same = 1;
    end

end
