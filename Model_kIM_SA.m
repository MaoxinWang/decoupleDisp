function [median_kIM,sigma_lnkIM] = Model_kIM_SA(IM1,IM2,Ts,h_ratio,IR,kIM)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or wangmx@whu.edu.cn)
% May 2024
%
% SA-dependent predictive models for equivalent loading parameters k-IMs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   IM1      = first predictor intensity measure (PGA or ASI)
%   IM2      = second predictor intensity measure
%  (IM2 = SA when kIM is k-PGA or k-ASI; IM2 = IM when kIM is k-IM other than k-PGA and k-ASI)
%   IM3      = third predictor intensity measure
%  (IM3 = SA when kIM is k-IM other than k-PGA and k-ASI; otherwise IM3 is not needed)
%   Ts       = natural period of full soil depositt (units: s)
%   h_ratio  = ratio of the sliding depth h to the soil deposit thickness H
%   IR       = soil-bedrock impedance ratio
%   (Note: the above inputs can be scalar or matrix and must be in the same dimension)
%   kIM      = target k-IM (k-PGA, k-PGV, k-IA, k-SI, k-CAV, or k-ASI)
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
x_Ts = log(reshape(Ts,n_data,1));
x_h_ratio = reshape(h_ratio,n_data,1);
x_IR = reshape(IR,n_data,1);

%% specify model parameters
% coefficients for calculating standard deviation
coff_SigmaALL = [
    0.201 	0.023 	0.000 	0.002 	-0.017 	0.011
    0.131 	0.000 	-0.038 	-0.008 	-0.040 	0.000
    0.326 	0.050 	0.000 	0.002 	-0.040 	0.017
    0.047 	0.000 	-0.017 	-0.004 	-0.037 	-0.010
    0.174 	0.038 	0.000 	0.000 	-0.017 	0.008
    0.121 	0.032 	0.000 	0.000 	-0.024 	0.000
    ];

% neural network weight and bias coefficients
switch kIM
    case 'k-PGA'
        IM_raw = exp(x_IM1);
        X_min = [-5.84889 	-6.48306 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	0.95143 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_Ts, x_h_ratio, x_IR];  % [PGA, SA(1s),...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        ind_coefSigma = 1;
        weightMatrix = [
            0.53764 	1.15916 	0.01144 	0.08663 	-1.49284 	-0.35933 	0.90536 	1.67709 	-1.05703 	1.43464 	0.17759 	-0.08562 	-0.27140 	0.57315 	1.87035 	-1.00478
            -0.75570 	-1.31119 	-0.09915 	0.10265 	1.47542 	0.65151 	-0.39764 	0.12020 	1.28933 	0.79700 	0.45216 	0.03782 	0.16544 	-0.84813 	-1.77889 	1.09760
            0.89644 	1.46656 	0.49092 	-0.10258 	-1.17773 	-0.62774 	0.58720 	1.25156 	-1.41669 	0.51456 	-0.11282 	-0.70478 	-1.02846 	0.41609 	0.95874 	-1.08159
            1.32754 	-0.14236 	1.67413 	-0.01002 	0.08658 	-1.55390 	0.08672 	-0.15540 	0.14693 	0.00076 	0.01187 	-1.36827 	0.01873 	-0.40827 	0.52847 	-0.35493
            -0.01240 	-0.20024 	-0.00439 	1.21416 	-0.11110 	0.00451 	-0.07177 	-1.73475 	0.38468 	-0.11633 	-0.14609 	0.00817 	-0.06867 	0.05141 	0.00145 	-0.09602
            ];
        weightVector = [
            4.08126 	3.36380 	3.38991 	0.64170 	1.39897 	4.10783 	3.30912 	0.11442 	2.10136 	0.42618 	3.10367 	3.57170 	2.25713 	2.45220 	1.58028 	2.92534
            ];
        biasVector = [
            0.27570 	0.31765 	0.27397 	0.02036 	0.16544 	-0.24348 	0.40683 	-1.11570 	-0.46486 	-0.60228 	0.60026 	-0.26275 	-0.14991 	-1.82130 	-2.32785 	1.75795
            ];
        biasScalar = -4.08066;
    case 'k-PGV'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-1.53247 	-8.23853 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	4.89336 	0.17587 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, PGV, SA(2s),...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        ind_coefSigma = 2;
        weightMatrix = [
            -0.51589 	-0.47815 	0.88781 	-0.30777 	0.27217 	0.13090 	1.06094 	-1.91526 	4.17829 	0.30103 	4.47523 	-0.33325 	0.54910 	-0.20967 	-1.18357 	0.17372 	1.94490 	-4.20830
            0.39166 	-0.70511 	-1.24929 	-0.06734 	0.61073 	-0.05061 	-1.52559 	2.94444 	-4.39391 	0.64206 	-1.98469 	0.24190 	0.72897 	0.98127 	0.67143 	0.24908 	-1.98134 	4.29746
            0.31258 	1.66202 	0.66620 	0.41788 	-0.41906 	-0.24097 	0.68594 	-1.76555 	0.55611 	-0.35455 	-1.03831 	0.26935 	-1.75504 	-0.91956 	0.66574 	-0.31791 	0.18762 	-0.49719
            -0.46021 	-1.28328 	0.39182 	-0.72104 	0.29776 	0.80068 	0.33667 	-0.15780 	2.06650 	0.31181 	2.73074 	-0.32909 	0.97737 	-0.90703 	-1.09207 	0.58982 	1.25471 	-2.17628
            0.05960 	0.15213 	-0.23248 	0.12169 	0.02662 	-0.01929 	0.34271 	-0.03529 	0.00337 	0.02357 	0.04107 	-0.38419 	0.09651 	0.04173 	-0.47941 	-0.09809 	0.77821 	0.00050
            0.28183 	-0.04510 	-0.07567 	-0.31812 	-0.16647 	-0.13790 	-0.10286 	0.03503 	-0.19722 	-0.17308 	-0.32909 	0.03175 	0.05951 	0.32075 	-0.00213 	0.65626 	-0.02985 	0.14448
            ];
        weightVector = [
            4.19410 	3.05733 	3.10263 	3.52873 	3.86017 	5.20065 	2.48451 	1.35585 	2.54072 	2.77923 	0.16510 	2.34719 	4.00321 	1.52553 	7.35157 	1.61891 	3.74549 	2.69303
            ];
        biasVector = [
            0.06369 	1.49785 	-0.62042 	0.25657 	0.85908 	-0.69247 	-0.38738 	0.51966 	-1.00378 	-0.48095 	-0.71632 	0.34959 	-1.27992 	0.63556 	1.82728 	-0.29463 	-2.34592 	1.07604
            ];
        biasScalar = 0.38250;
    case 'k-IA'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-8.48941 	-6.48306 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	3.01098 	0.95143 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, IA, SA(1s),...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        ind_coefSigma = 3;
        weightMatrix = [
            -0.11886 	1.54693 	0.05647 	-0.39309 	0.16642 	-0.43378 	-0.19935 	0.14510 	0.25006 	-0.69799 	0.11350 	-0.31544 	0.24509 	-0.02614 	0.38684 	-1.49410 	-0.00025 	0.54633
            1.03258 	0.80040 	-1.62888 	0.04105 	0.18832 	0.18432 	-1.02148 	0.82110 	0.25749 	2.87586 	-0.86219 	0.15443 	0.07362 	-1.48273 	0.60224 	-0.51350 	0.66150 	0.03640
            -0.34239 	-2.03970 	1.82871 	0.47669 	0.25039 	0.26570 	1.51395 	-1.42230 	0.29140 	-2.53150 	0.17458 	0.08058 	-0.19974 	1.82504 	-1.16727 	1.91712 	-0.82452 	-0.73258
            2.49963 	1.71440 	0.78919 	-0.61250 	0.10044 	-0.44192 	-1.26613 	1.07080 	0.27655 	-0.70398 	-2.37189 	-0.36159 	0.36436 	-1.48847 	-0.85713 	-1.56375 	0.78842 	0.78394
            -0.07990 	-0.03110 	0.28307 	0.06541 	-0.10830 	0.22232 	-1.32239 	-0.58603 	-0.07836 	-0.25546 	0.07959 	-0.38994 	-0.08344 	-1.04504 	-0.30784 	0.02823 	1.01582 	-0.06494
            0.42625 	-0.18359 	0.00998 	0.44189 	-0.19378 	-0.11601 	-0.03126 	-0.00192 	-0.25747 	0.00489 	-0.44864 	-0.03861 	0.70259 	-0.05349 	-0.02615 	0.10639 	0.03697 	-0.19098
            ];
        weightVector = [
            4.78448 	4.38370 	13.97003 	10.55880 	10.02628 	13.27007 	4.02476 	11.71620 	5.48550 	5.35502 	5.03109 	13.11221 	3.55914 	3.82669 	10.40314 	6.72210 	9.47134 	18.37944
            ];
        biasVector = [
            1.36020 	-0.23600 	-0.19355 	0.12092 	0.88922 	0.42739 	1.00780 	-3.20321 	-0.38927 	0.22641 	-1.27683 	1.04165 	-0.22153 	2.08074 	0.15208 	0.36578 	-1.01759 	-0.23200
            ];
        biasScalar = -8.31340;
    case 'k-SI'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-0.34321 	-8.23853 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	6.54000 	0.17587 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, SI, SA(2s),...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        ind_coefSigma = 4;
        weightMatrix = [
            -0.27031 	0.07341 	-0.09120 	-0.34425 	2.36646 	-0.15372 	-0.22427 	-2.53015 	0.78935 	-2.37062 	0.20904 	-0.15857 	0.32969 	-0.10367 	0.10405 	-0.46646 	-0.36063 	0.69164
            0.72426 	-1.84411 	1.70676 	-0.70313 	-0.80155 	0.67998 	1.20355 	1.26861 	-0.05591 	5.00407 	-0.93547 	2.28157 	0.73234 	0.79785 	-0.52373 	0.03977 	0.05404 	2.33503
            -0.47544 	2.17787 	-0.03781 	1.26735 	-0.62397 	0.35525 	-1.00097 	0.31215 	-0.44732 	-2.30154 	0.83796 	-2.65645 	-1.20883 	0.04190 	0.40625 	0.11655 	0.70045 	-3.67038
            0.25266 	-1.47945 	1.57436 	-0.58484 	2.55017 	0.42871 	1.62332 	-2.64622 	-0.63462 	-2.69508 	-0.53724 	1.83292 	0.07729 	0.56052 	0.49963 	0.70832 	-0.92041 	0.76879
            -0.44724 	0.16918 	0.06077 	-0.31575 	0.06499 	-0.02663 	0.27681 	-0.07768 	-0.00880 	-0.06215 	0.39815 	-0.10797 	0.03321 	0.01512 	0.20009 	-0.00231 	-0.10614 	0.23529
            0.00658 	0.02616 	-0.38609 	0.03104 	-0.28181 	-0.16452 	0.19639 	0.26348 	1.16328 	-0.78372 	-0.01735 	-0.14226 	0.08663 	-0.15539 	0.07523 	-1.15331 	-0.04443 	0.01778
            ];
        weightVector = [
            4.31981 	6.27036 	0.72386 	6.62452 	2.01196 	2.50239 	0.51340 	1.92198 	1.65686 	0.10897 	3.65970 	3.47420 	4.70797 	2.75135 	3.68035 	2.81840 	2.95633 	1.56916
            ];
        biasVector = [
            -0.21667 	1.28019 	-1.61905 	1.23640 	-1.22035 	1.33284 	-0.85536 	1.34560 	2.24257 	1.60431 	-0.03387 	-1.35148 	0.14179 	0.02285 	0.50598 	-2.42210 	-0.85824 	-1.33915
            ];
        biasScalar = -2.81365;
    case 'k-CAV'
        IM_raw = exp(x_IM2);
        X_min = [-5.84889 	-4.10006 	-6.48306 	-3.91202 	0.00000 	0.10000];
        X_max = [0.46052 	1.59938 	0.95143 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_IM3, x_Ts, x_h_ratio, x_IR];  % [PGA, PGV, SA(1s),...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        ind_coefSigma = 5;
        weightMatrix = [
            -0.50275 	-0.75143 	-0.20674 	0.45954 	0.78371 	-0.27460 	-0.48431 	0.62725 	0.42832 	1.55160 	0.46512 	-0.25639 	0.56874 	0.84243 	0.27935 	1.28228 	0.40764 	-1.25159
            -0.53401 	-1.44727 	0.00238 	0.13862 	0.08054 	0.19410 	-0.33137 	0.16995 	0.60444 	0.68238 	0.59988 	-0.31773 	0.20478 	2.26940 	0.12890 	0.02938 	-0.39861 	-0.34874
            1.31727 	2.35188 	0.30535 	-0.48679 	-0.59931 	0.03755 	0.69284 	-1.08371 	0.55621 	-1.90433 	-1.31352 	1.12299 	-0.96915 	-3.16014 	0.01136 	-1.35392 	-0.23167 	1.62676
            -1.28593 	1.01983 	-0.31782 	-2.58119 	0.84669 	-0.21758 	2.66014 	1.01949 	1.01868 	1.70573 	1.46544 	-0.78213 	0.76508 	-0.96329 	0.36787 	1.33539 	-0.98028 	-1.47783
            -1.55580 	0.18010 	-0.00237 	-2.98174 	-0.00982 	0.00750 	3.09147 	-0.05043 	-0.08511 	-0.02895 	1.59540 	0.02330 	-0.93665 	-0.17091 	-0.02511 	0.66546 	-0.16653 	0.09563
            -0.01163 	0.07990 	0.44761 	0.06259 	1.45382 	-0.14815 	-0.07031 	0.03045 	-0.19976 	-0.05184 	0.00566 	-0.17335 	0.05604 	-0.07690 	-0.08921 	-0.07235 	-0.02918 	-0.07218
            ];
        weightVector = [
            2.68324 	1.40810 	5.22355 	2.96711 	0.43204 	12.64647 	2.60969 	5.85305 	0.51482 	1.39614 	3.11927 	2.40985 	5.65132 	0.88992 	7.24649 	0.65655 	2.00584 	3.24485
            ];
        biasVector = [
            0.03579 	-0.41584 	-0.10059 	-0.10906 	-0.06683 	0.16432 	0.10034 	-0.21052 	-0.80486 	0.02017 	-0.00082 	0.43510 	-3.13278 	0.54887 	0.36992 	1.14453 	0.01983 	0.22628
            ];
        biasScalar = -0.09007;
    case 'k-ASI'
        IM_raw = exp(x_IM1);
        X_min = [-6.00510 	-5.72466 	-3.91202 	0.00000 	0.10000];
        X_max = [0.06176 	0.89774 	0.69315 	1.00000 	1.00000];
        X_Input = [x_IM1, x_IM2, x_Ts, x_h_ratio, x_IR];  % [ASI, SA(0.5s),...]
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        ind_coefSigma = 6;
        weightMatrix = [
            1.68269 	-1.19586 	0.31454 	0.00198 	1.06852 	-1.75117 	1.11189 	0.22415 	-1.10503 	0.05436 	-0.85843 	-0.28014 	0.16619 	1.18890 	-0.28367 	-1.84168
            -1.64736 	-0.05462 	0.21578 	0.00843 	0.35428 	1.41530 	-1.33670 	-0.40215 	1.21529 	-0.31362 	1.89788 	0.32943 	0.07568 	-0.27079 	0.53576 	1.71088
            2.45551 	-1.95939 	0.28782 	0.18266 	1.37574 	-2.39112 	1.71994 	-0.33340 	-1.82381 	1.47052 	-2.53196 	-0.61347 	-0.05002 	2.73852 	-1.51351 	-2.50474
            -0.06106 	-0.01208 	0.00309 	0.66300 	0.00542 	0.08317 	1.15983 	-0.04357 	-1.49994 	3.36815 	0.30447 	-0.28988 	-0.01669 	-0.04257 	-2.93466 	0.10772
            -0.07838 	0.27610 	-0.10475 	0.08771 	-0.28347 	1.65054 	-0.01737 	-0.00885 	0.08219 	-0.06165 	-0.26069 	-0.20086 	0.84658 	-0.23982 	0.04907 	-0.02289
            ];
        weightVector = [
            3.88083 	3.36116 	6.83906 	2.13568 	2.55573 	0.19782 	0.83366 	4.71300 	1.10940 	1.46453 	0.21130 	3.41655 	1.04973 	1.41119 	1.86862 	3.48167
            ];
        biasVector = [
            -0.17379 	1.41162 	0.39997 	-0.82777 	-1.01307 	-0.37637 	-0.58087 	-0.09176 	1.34390 	-0.17603 	0.94218 	0.94900 	-0.11471 	-2.03508 	0.16910 	0.27043
            ];
        biasScalar = -6.34647;
end

%% calculate median and standard deviation
% median
X_norm = (1-(-1))*(X_Input-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;
lnkIM = (2./(1+exp(-2*(X_norm*weightMatrix+repmat(biasVector,[n_data,1]))))-1)*weightVector'+biasScalar;
median_kIM = exp(lnkIM);

% standard deviation
coff_SigmaModel = coff_SigmaALL(ind_coefSigma,:);
lnTs = X_Input(:,end-2);
lnIR = log(X_Input(:,end));
sigma_lnkIM = coff_SigmaModel(1)+coff_SigmaModel(2)*lnTs++coff_SigmaModel(3)*lnTs.^2+...
    coff_SigmaModel(4)*lnTs.^3+coff_SigmaModel(5)*lnIR+...
    coff_SigmaModel(6)*lnTs.*lnIR;

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
