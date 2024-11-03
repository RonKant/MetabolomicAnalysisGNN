function metadata = constructmetadata(mzxmlDirectory)
    files = dir(mzxmlDirectory);

    metadata = table();
    metadata.fileName = strings(0);
    metadata.ionization = strings(0);
    metadata.drugName = strings(0);
    metadata.time = strings(0);
    metadata.concentration = strings(0);

    for file = files'

        if file.isdir == 1
            continue
        end

        filenameParts = split(file.name, "_");

        newRow = array2table([ ...
                                  file.name ...
                                  string(filenameParts(1)) ...
                                  string(filenameParts(2)) ...
                                  string(filenameParts(3)) ...
                                  string(filenameParts(4)) ...
                              ], 'variablenames', metadata.Properties.VariableNames);

        metadata = [metadata; newRow];
    end

    metadata.isDrug = ((metadata.drugName ~= 'Blank') & (metadata.drugName ~= 'QC'));
end
