function [median_kIM,sigma_lnkIM] = Model_kIM_Tm(IM1,IM2,IM3,Ts,h_ratio,IR,kIM)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or wangmx@whu.edu.cn)
% May 2024
%
% Tm-dependent predictive models for equivalent loading parameters k-IMs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   IM1      = first predictor intensity measure (PGA or ASI)
%   IM2      = second predictor intensity measure
%  (IM2 = Tm when kIM is k-PGA£¬ k-Tm or k-ASI; IM2 = IM when kIM is k-IM other than k-PGA and k-ASI)
%   IM3      = third predictor intensity measure
%  (IM3 = Tm when kIM is k-IM other than k-PGA and k-ASI; otherwise IM3 is not needed)
%   Ts       = natural period of full soil deposit (units: s)
%   h_ratio  = ratio of the sliding depth h to the soil deposit thickness H
%   IR       = soil-bedrock impedance ratio
%   (Note: the above inputs can be scalar or matrix and must be in the same dimension)
%   kIM      = target k-IM (k-PGA, k-PGV, k-Tm, k-IA, k-SI, k-CAV, or k-ASI)
%
% OUTPUT
%
%   median_kIM   = median prediction of k-IM
%   sigma_lnkIM  = standard deviation of ln(k-IM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% deal with input data
[n_row,n_col] = size(IM1);
n_data = n_row*n_col;
x_IM1 = log(reshape(IM1,n_data,1));
x_IM2 = log(reshape(IM2,n_data,1));
try
    x_IM3 = log(reshape(IM3,n_data,1));
catch
    % nothing
end
x_Ts = log(reshape(Ts,n_data,1));
x_h_ratio = reshape(h_ratio,n_data,1);
x_IR = reshape(IR,n_data,1);

%% specify model parameters
% coefficients for calculating standard deviation
coff_SigmaALL = [
    0.1578 	0.0275 	-0.0044 	0.0026 	-0.0311 	0.0000 	0.0121
    0.1212 	0.0544 	0.0000 	0.0000 	-0.0299 	0.0000 	0.0000
    0.1286 	0.0680 	0.0138 	-0.0025 	-0.0136 	0.0000 	0.0058
    0.1678 	0.0299 	-0.0027 	0.0042 	-0.0499 	0.0102 	0.0145
    0.0521 	0.0216 	0.0000 	0.0000 	-0.0279 	0.0000 	-0.0069
    0.1000 	0.0224 	0.0000 	0.0017 	-0.0152 	0.0069 	0.0062
    0.0994 	0.0155 	-0.0049 	0.0016 	-0.0236 	0.0000 	0.0030
    ];

% range for calculating standard deviation
min_lnTsTm = -2.5;
max_lnTsTm = 3;

% neural network weight and bias coefficients
switch kIM
    case 'k-PGA'
        IM_raw = exp(x_IM1);
        X_min = [-5.84889 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_Ts, x_h_ratio, x_IR];  % [PGA, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 1;
        weightMatrix = [
            0.14132 	0.06968 	0.72642 	-0.55089 	-0.05082 	-0.02452 	-0.03422 	1.18518 	0.04955 	0.56182 	0.09091 	0.81592 	1.02872 	0.15359
            0.71930 	0.60642 	0.94020 	0.88685 	-0.70257 	0.74192 	-0.22642 	-0.02839 	0.38386 	0.07279 	-0.16312 	-0.06684 	-1.06598 	-0.90012
            -0.97746 	-0.89962 	-1.07019 	-1.45833 	0.99100 	-1.06881 	0.39129 	0.62846 	-0.60986 	0.01227 	-0.49117 	0.50739 	1.76969 	1.34781
            -0.75257 	-1.13651 	0.00802 	-0.03220 	0.79058 	0.08127 	1.14909 	-0.02944 	-1.22389 	-0.01184 	-0.02437 	0.02269 	0.07391 	-0.07327
            -0.14575 	-0.00305 	-0.24855 	0.07844 	0.01024 	0.52188 	-0.00159 	-0.23151 	-0.01388 	0.84897 	-0.10456 	-0.15186 	-0.14009 	-0.27532
            ];
        weightVector = [
            1.47696 	4.62542 	0.61969 	4.66569 	4.54968 	1.81090 	4.26000 	1.48018 	3.48193 	0.54270 	1.86138 	2.93637 	1.69769 	3.27524
            ];
        biasVector = [
            1.89438 	0.44254 	0.41647 	0.04787 	-0.11234 	-0.39672 	-0.07889 	-0.45089 	-0.31053 	-0.17887 	0.05969 	0.81703 	0.04257 	0.20041
            ];
        biasScalar = -5.78344;
    case 'k-PGV'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-1.53247 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	4.89336 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, PGV, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 2;
        weightMatrix = [
            -1.36321 	-2.58809 	0.88827 	-1.56368 	-2.52830 	-4.54394 	-0.74306 	1.28196 	0.48742 	-0.75391 	1.18835 	-0.29598
            1.32526 	2.53505 	-0.18266 	1.36980 	2.32777 	4.00714 	0.34058 	-1.14541 	0.05893 	0.70497 	-1.06517 	0.47740
            0.26567 	-2.49074 	0.31841 	-1.63211 	-1.61083 	-1.45492 	1.42913 	1.53381 	0.13445 	0.67379 	-0.53710 	-0.06103
            -1.23989 	1.35003 	0.28786 	0.99039 	0.25271 	-1.48248 	-2.13725 	-1.05863 	0.19713 	-1.36748 	1.41837 	-0.15007
            0.05772 	0.42862 	-0.01815 	1.08564 	-0.20728 	-0.50783 	0.00643 	-0.71154 	-0.01582 	0.48047 	-0.04212 	0.02139
            0.57587 	0.14359 	-0.34269 	0.04860 	-0.14409 	-0.05720 	0.40360 	-0.05634 	-0.22447 	-0.12084 	-0.43701 	0.21335
            ];
        weightVector = [
            2.29837 	1.28528 	2.32345 	1.58588 	0.33337 	0.38635 	0.89837 	3.40278 	3.79132 	0.55141 	3.69565 	4.81856
            ];
        biasVector = [
            0.38005 	-1.71684 	-0.64782 	-1.44136 	0.06979 	1.80994 	0.94951 	1.31947 	0.70423 	0.79303 	-0.53005 	-0.28845
            ];
        biasScalar = 1.39261;
    case 'k-Tm'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_Ts, x_h_ratio, x_IR];  % [PGA, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 3;
        weightMatrix = [
            -0.20028 	0.27911 	-0.54940 	0.20027 	-1.26250 	-0.28452 	-0.11792 	-0.62008 	-0.41039 	-0.06076 	-0.19728 	-0.03941 	-0.17923 	0.12085 	0.10138 	0.31507
            -0.82273 	1.20963 	-0.10737 	0.12192 	-2.75417 	-0.81291 	0.63482 	0.81087 	-0.82597 	-0.03830 	-1.56745 	0.14534 	-1.02059 	-0.35426 	0.40148 	-0.96001
            1.17605 	-1.41720 	-0.06446 	0.05236 	1.87796 	1.19866 	-1.26953 	-1.75730 	2.63183 	-0.02206 	2.18187 	-0.09080 	1.39895 	1.08831 	0.61856 	1.75884
            -0.01801 	2.26158 	-0.01075 	0.01042 	0.15373 	-0.01921 	-0.23849 	0.00812 	7.13902 	-0.53641 	2.34297 	-0.47091 	-0.01283 	0.28834 	-0.04080 	0.06307
            -0.64989 	0.00531 	0.11527 	-0.08020 	0.12215 	-0.99793 	-0.05553 	0.22225 	0.11079 	0.33383 	-0.02791 	0.36409 	-0.47438 	0.05172 	0.30230 	-0.06502
            ];
        weightVector = [
            -7.15107 	5.03492 	3.71653 	10.17799 	0.62227 	2.23142 	7.66021 	1.84475 	0.05378 	2.96942 	0.22173 	-2.83171 	4.41713 	5.27774 	0.72397 	4.15086
            ];
        biasVector = [
            -0.33573 	5.53099 	-0.18247 	-0.00136 	-3.91290 	-0.28075 	0.04296 	-0.38345 	2.58781 	0.64326 	-1.06769 	0.57179 	-0.50365 	-0.17075 	0.37842 	0.18546
            ];
        biasScalar = -4.41350;
    case 'k-IA'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-8.48941 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	3.01098 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, IA, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 4;
        weightMatrix = [
            0.99986 	-0.02528 	0.27882 	0.08530 	0.70050 	-0.01965 	-1.03084 	0.49313 	-0.12067 	-0.16879 	0.20469 	-0.48550 	-0.22931 	-0.37847 	1.60634 	0.65300 	-0.43928 	0.18770
            -0.49034 	0.05273 	0.09659 	0.21746 	-0.48881 	-0.05976 	0.53166 	-0.05720 	-0.04823 	0.96368 	-0.80359 	0.43193 	0.07100 	0.41787 	-0.31204 	0.53550 	0.06892 	-0.03116
            -1.28634 	0.28665 	-0.30510 	0.61386 	-0.42142 	-1.21483 	1.22341 	1.22880 	-1.09668 	-1.94323 	1.76827 	0.32886 	-0.97543 	-0.07482 	-0.45484 	-0.22158 	-1.22578 	1.03633
            3.33236 	-0.41918 	0.50554 	-0.45123 	1.06750 	1.34654 	-3.34758 	-1.46026 	1.05878 	2.48006 	-2.24567 	-0.79892 	1.10848 	-0.08253 	1.78662 	0.73585 	1.45647 	-1.11901
            0.09384 	-0.02644 	0.06236 	-0.20965 	-0.09011 	-1.45921 	-0.10277 	0.11862 	0.71465 	-0.06456 	0.09704 	0.06525 	1.20935 	-0.00450 	-0.23626 	-0.02847 	-0.07320 	-0.98324
            0.00100 	0.25363 	-0.11464 	-0.08437 	3.46104 	0.08129 	-0.03545 	-0.18138 	0.03706 	-0.03554 	-0.03559 	-2.33580 	0.02845 	-0.05648 	-5.93452 	-0.26098 	0.11898 	-0.02655
            ];
        weightVector = [
            6.08195 	10.74880 	11.15033 	7.16360 	2.06533 	15.97811 	5.97684 	6.67079 	14.21169 	3.95135 	5.19895 	2.76301 	12.38564 	8.84106 	0.21905 	1.47515 	8.56578 	25.22257
            ];
        biasVector = [
            0.77948 	-0.40571 	0.74530 	0.61085 	0.92735 	-4.90713 	-0.74102 	0.16041 	-0.83178 	0.05305 	0.02736 	-0.51425 	-1.11490 	0.41241 	-4.51680 	-0.37634 	-0.08157 	0.98928
            ];
        biasScalar = 3.81218;
    case 'k-SI'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-0.34321 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	6.54000 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, SI, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 5;
        weightMatrix = [
            -1.50426 	0.02264 	0.28368 	-0.21776 	1.85997 	0.29610 	-0.00162 	0.35753 	-1.25589 	0.14238 	0.50764 	0.09964 	-0.14308 	-0.64793 	-0.80280 	1.38972
            1.77580 	2.13381 	0.06733 	1.11849 	-2.28234 	-0.16235 	0.42188 	-0.39566 	1.55202 	-0.42831 	-1.38920 	-0.53537 	0.22806 	0.42518 	0.88655 	-1.69725
            -0.67715 	0.04518 	-1.15371 	-0.30185 	1.70222 	0.21687 	-0.15209 	0.93178 	-1.22216 	0.50502 	0.48365 	1.47301 	-0.09102 	0.75095 	-1.27706 	0.68376
            -1.56918 	2.06097 	1.66009 	1.45276 	0.70721 	0.02089 	0.62211 	-1.26250 	-0.47557 	-0.76788 	-1.63201 	-1.68452 	-0.14399 	-1.57715 	1.17740 	1.56459
            0.05386 	0.04719 	-0.01507 	0.03743 	0.14823 	-0.00420 	0.00367 	-0.62688 	-0.10231 	0.03100 	-0.04158 	-0.00496 	0.00739 	0.03473 	0.72007 	0.04004
            -0.20255 	-0.45372 	-0.31178 	-0.26288 	0.06534 	-0.17570 	-0.04422 	-0.04632 	-0.05725 	0.02775 	0.33093 	0.17523 	0.38734 	0.47403 	0.04833 	0.10238
            ];
        weightVector = [
            2.95889 	0.37823 	4.67028 	3.25159 	2.69597 	6.83408 	6.71597 	4.92230 	5.50112 	3.97353 	2.15767 	3.02719 	4.04076 	2.01429 	3.72309 	3.32005
            ];
        biasVector = [
            1.44414 	-2.04421 	-0.34550 	-0.33983 	-0.33331 	0.14451 	0.76460 	1.58391 	0.29311 	-0.73307 	0.32176 	0.36167 	-0.11837 	0.35610 	-1.58082 	-1.37098
            ];
        biasScalar = -0.14309;
    case 'k-CAV'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-4.10006 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	1.59938 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, CAV, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 6;
        weightMatrix = [
            0.01346 	-0.06460 	0.23929 	-0.19530 	-0.00359 	0.27146 	-0.54723 	0.71752 	-0.22538 	0.06960 	-0.07076 	-0.31639 	0.58724 	-0.29425
            0.12550 	0.33878 	-0.15821 	0.11355 	-0.26902 	-0.11667 	-0.15558 	-0.19606 	-0.11421 	0.23067 	0.29751 	0.20406 	0.37335 	0.10275
            -0.00206 	-0.01937 	-0.27180 	0.57471 	0.88659 	1.05972 	0.43838 	1.05120 	1.34616 	-1.32836 	-0.14629 	0.00423 	-0.15451 	-1.02251
            -0.00779 	0.05386 	-0.08221 	-0.13358 	-1.30889 	-1.24132 	-0.68408 	-0.82104 	-2.79986 	2.25480 	0.25860 	0.32775 	0.39150 	1.02889
            -0.00326 	-0.01055 	-1.95372 	1.98795 	0.04283 	-0.91812 	-0.02560 	-0.19002 	-0.00313 	-0.01530 	-0.01484 	2.15670 	-0.04045 	0.52824
            0.41014 	-0.39324 	0.00891 	-0.01597 	-0.14217 	-0.01908 	0.10608 	-0.18436 	-0.20671 	0.13969 	-0.66163 	0.00751 	-0.21219 	0.07222
            ];
        weightVector = [
            7.54826 	6.59049 	7.58077 	2.93620 	3.08799 	2.10387 	3.12417 	1.54958 	1.60474 	3.38832 	3.83537 	4.07212 	1.57534 	3.66498
            ];
        biasVector = [
            0.07127 	0.65769 	-1.98954 	1.99467 	-0.23398 	0.76683 	1.03277 	0.55208 	-0.55665 	0.53886 	-0.99200 	2.18090 	-0.43618 	-0.47886
            ];
        biasScalar = -4.33677;
    case 'k-ASI'
        IM_raw = exp(x_IM1);
        X_min = [-6.00510 	-2.70079 	-3.91202 	0.00000 	0.10000];
        X_max = [0.06176 	0.50155 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_Ts, x_h_ratio, x_IR];  % [ASI, Tm,...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        lnTsTm = log(exp(X_Input(:,end-2))./exp(X_Input(:,end-3)));
        lnTsTm(lnTsTm<min_lnTsTm) = min_lnTsTm;
        lnTsTm(lnTsTm>max_lnTsTm) = max_lnTsTm;
        ind_coefSigma = 7;
        weightMatrix = [
            -0.19385 	0.48457 	0.10019 	-0.05502 	-0.05091 	0.66369 	0.23265 	0.00277 	-0.01690 	-0.06310 	-0.43684 	1.32650 	-0.25305 	-0.54140 	0.20075 	0.59940 	0.07483 	0.05479
            -0.49542 	0.76725 	-0.01599 	-0.15038 	-0.15550 	-0.29659 	-0.52322 	0.04144 	0.37371 	-0.64999 	0.36047 	0.15246 	-0.57530 	-0.12664 	0.42966 	0.12276 	0.53467 	0.15301
            0.00555 	-1.18518 	0.26489 	0.74813 	-0.70647 	2.27503 	-1.40391 	-0.37407 	-2.59744 	1.98433 	-2.09813 	0.96286 	1.34796 	-9.09337 	0.40042 	0.70889 	-1.62124 	-1.24491
            0.74208 	-0.22121 	0.01395 	1.32986 	-0.34352 	0.02113 	-0.22709 	-0.02993 	-2.33904 	0.01988 	0.00715 	0.04979 	0.40352 	-3.67543 	-0.45786 	0.04505 	0.00768 	-1.66694
            0.00535 	-0.14678 	0.81979 	-0.00505 	0.04677 	-0.06945 	0.01358 	-0.54346 	0.07059 	-0.22355 	-0.00105 	-0.24820 	0.11182 	-0.06897 	-0.01115 	-0.11406 	0.45428 	0.00051
            ];
        weightVector = [
            4.45749 	1.76230 	2.57867 	3.20068 	4.56044 	4.44923 	1.23455 	3.76307 	0.97863 	2.92167 	6.59537 	1.08940 	2.76059 	0.20096 	5.87778 	4.91413 	2.01793 	1.92235
            ];
        biasVector = [
            1.10047 	0.37147 	-0.22646 	-0.37814 	-0.55163 	0.03893 	1.13321 	0.30960 	1.74935 	0.19186 	0.05524 	-0.54925 	-0.32581 	7.22200 	-0.91420 	0.86659 	-0.30600 	0.15090
            ];
        biasScalar = -4.60583;
end

%% calculate median and standard deviation
% median
X_norm = (1-(-1))*(X_Input-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;
lnkIM = (2./(1+exp(-2*(X_norm*weightMatrix+repmat(biasVector,[n_data,1]))))-1)*weightVector'+biasScalar;
median_kIM = exp(lnkIM);

% standard deviation
coff_SigmaModel = coff_SigmaALL(ind_coefSigma,:);
lnIR = log(X_Input(:,end));
sigma_lnkIM = coff_SigmaModel(1)+coff_SigmaModel(2)*lnTsTm++coff_SigmaModel(3)*lnTsTm.^2+...
    coff_SigmaModel(4)*lnTsTm.^3+coff_SigmaModel(5)*lnIR+coff_SigmaModel(6)*lnIR.^2+...
    coff_SigmaModel(7)*lnTsTm.*lnIR;

%% modify predictions for Ts<0.02 s
ind_TsOut = (exp(x_Ts)<0.02);
n_TsOut = sum(ind_TsOut);
if n_TsOut>0
    median_TsOut = median_kIM(ind_TsOut);
    k_line1 = (median_TsOut-IM_raw(ind_TsOut))/0.02;
    median_kIM(ind_TsOut) = IM_raw(ind_TsOut)+k_line1.*exp(x_Ts(ind_TsOut));
    
    sigma_TsOut = sigma_lnkIM(ind_TsOut);
    k_line2 = (sigma_TsOut-0)/0.02;
    sigma_lnkIM(ind_TsOut) = IM_raw(ind_TsOut)+k_line2.*exp(x_Ts(ind_TsOut));
end
median_kIM = reshape(median_kIM,n_row,n_col);
sigma_lnkIM = reshape(sigma_lnkIM,n_row,n_col);
