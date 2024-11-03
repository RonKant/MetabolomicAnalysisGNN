global CONFIG
CONFIG = {};

%% raw data analysis
% EIC extraction
Config.MS_ACCURACY = 5/1000000; % 5 ppm
Config.MZ_SLICING_WIDTH = 0.0001;
Config.MAX_MZ = 1200;
% Config.MIN_MZ = 70;
Config.EIC_CONSECUTIVE_SCANS = 1;

% Peak detection
Config.BASELING_SMOOTH_WINDOW = 20;
Config.EIC_SMOOTH_ORDER = 3;
Config.EIC_SMOOTH_FRAMELEN = 7; % Should be odd integer
Config.MIN_SIGNAL_BASELINE_RATIO = 3;
Config.MIN_PEAK_INTENSITY = 1000;
Config.BASELINE_DROP_TOP_INTENSITY_PER = 50; %80;
Config.MIN_RT = 2; %12;
Config.LOCAL_BASELINE_SCANS = 5;
Config.MIN_SIGNAL_LOCAL_BASELINE_RATIO = 2;
Config.PEAK_FINDING_MERGE_MAX_RT = 20; % maximum RT distance.
Config.PEAK_FINDING_MERGE_INTENSITY_DIFF_PRCNT = 15;
% for peak clustering during peak finding
Config.PF_OVERSEGMENTATION_VALUE = 6; % max rt distance between peaks to be merged
Config.PF_PEAK_LOCATION_VALUE = 0.8; % 1 only peak, 0 all range - which values are used to calculate peak centroid.

% Peak grouping
Config.RT_TOL = 10; %10 seconds each side, for initial frouping
Config.RT_UNIQUETOL = 0.5; %0.5 sec difference between two groups will be consdiered as the same
Config.RT_TOL_AFTER_ALIGNMENT = 5; %seconds each side

% Peak alignment
Config.MIN_PEAKS_IN_GROUP = 2;
Config.MAX_NUM_GROUPS = 2000;
Config.POLY_DEGREE = 1;
Config.MAX_ALIGN_ITER = 3;
Config.ALIGN_CONVERGENCE_PRCTILE = 90;
Config.ALIGN_CONVERGENCE_TRESH = 1;

% Peak Filling
Config.PEAK_FILLING_MIN_FRAC = 0.7;
Config.PEAK_FILLING_NUM_SCANS = 3; % +- 3 scans from the closest RT

% High Quality Matrix Filtering
Config.HQ_MAX_NUM_NANS_SAMPLE = 33;
Config.HQ_MIN_MEAN_INTENSITY = 10 ^ 4;
