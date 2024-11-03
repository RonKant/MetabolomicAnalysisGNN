function samples = applyPeakAlignmentToSample(samples, poly_arr)

    for s = 1:length(samples)
        cur_poly_arr = poly_arr{s};

        for p = 1:length(samples{s}.peaks)
            samples{s}.peaks{p}.rt_before_realignment = samples{s}.peaks{p}.rt;
            samples{s}.peaks{p}.rt = applyAlignmentToArray(samples{s}.peaks{p}.rt, cur_poly_arr);
        end

    end

end

function rt_aligned = applyAlignmentToArray(rt, poly_arr)
    rt_aligned = rt;

    for i = 1:length(poly_arr)
        p = poly_arr{i};
        rt_aligned = polyval(p, rt_aligned);
    end

end
