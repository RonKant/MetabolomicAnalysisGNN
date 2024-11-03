function sample = loadsampleLCMS(file_path)
    sample = loadMzXml(file_path, 'VERBOSE', 'F');
    sample = sliceMzs(sample);
    sample = extractEICs(sample);
    sample = findPeaks(sample);
