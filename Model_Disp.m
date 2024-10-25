function [median_D,sigma_lnD] = Model_Disp(IM1,IM2,IM3,Ky,Ts,h_ratio,IR,typeModel)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or wangmx@whu.edu.cn)
% May 2024
%
% Predictive models for sliding displacement
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   IM1       = first predictor intensity measure (PGA)
%   IM2       = second predictor intensity measure
%   IM3       = third predictor intensity measure
%   Ky        = yeild accelerationt (units: g)
%   Ts        = natural period of full soil depositt (units: s)
%   h_ratio   = ratio of the sliding depth h to the soil deposit thickness H
%   IR        = soil-bedrock impedance ratio
%   (Note: the above inputs can be scalar or matrix and must be in the same dimension)
%   typeModel = input IM-vector
%
% OUTPUT
%
%   median_D    = median prediction of D
%   sigma_lnD   = standard deviation of ln(D)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% deal with input data
[n_row,n_col] = size(IM1);
n_data = n_row*n_col;
x_lnIM1 = log(reshape(IM1,n_data,1));
x_lnIM2 = log(reshape(IM2,n_data,1));
try
    x_lnIM3 = log(reshape(IM3,n_data,1));
catch
    % nothing
end
x_lnKy = log(reshape(Ky,n_data,1));
x_Ts = reshape(Ts,n_data,1);
x_h_ratio = reshape(h_ratio,n_data,1);
x_IR = reshape(IR,n_data,1);

min_Ts = 0.0416;     max_Ts = 2;
min_h_ratio = 0.1; max_h_ratio = 1;
min_IR = 0.1;     max_IR = 1;
x_Ts(x_Ts<min_Ts & x_Ts>0) = min_Ts;
x_Ts(x_Ts>max_Ts) = max_Ts;
x_h_ratio(x_h_ratio<min_h_ratio) = min_h_ratio;
x_h_ratio(x_h_ratio>max_h_ratio) = max_h_ratio;
x_IR(x_IR<min_IR) = min_IR;
x_IR(x_IR>max_IR) = max_IR;

%% parameters of functional-form displacement models
% coefficients of the Saygili and Rathje (2008) model
coeff_D_SR08 = [
    -1.56 -4.58 -20.84 44.75 -30.5 -0.64 1.55;
    6.62 -3.93 -23.71 49.37 -32.94 0.93 1.79;
    2.39 -5.24 -18.78 42.01 -29.15 -1.56 1.38;
    ];

% coefficients of the standard deviation model
coeff_SigmaRigid = [
    0.41 0.52;
    0.60 0.26;
    0.46 0.56;
    0.372 0.505;
    0.155 0.737;
    0.301 0.617;
    0.184 0.753;
    0.269 0.655;
    ];
coeff_SigmaFlexi = [
    0.534 0.407 0;
    0.658 -0.203 0.372;
    0.586 0.590 0;
    0.487 0.395 0;
    0.178 0.619 0;
    0.324 0.547 0;
    0.216 0.678 0;
    0.290 0.622 0;
    ];

% coefficients of the standard deviation model
e_Mf = [
    -0.40560 	-1.68329 	2.75651 	-0.42977 	0.00000 	-2.60875 	0.41329 	-0.38873 	-0.89930 	0.46253 	0.30066 	0.00000 	0.10592
    0.92028 	-1.07255 	0.25957 	0.00000 	0.18423 	0.00000 	0.19963 	-0.17143 	-0.82143 	0.44338 	0.11239 	0.17476 	-0.10669
    -1.67374 	0.88757 	0.82602 	0.00000 	0.00000 	-2.20582 	1.22849 	-0.79659 	0.73397 	-0.37716 	0.81854 	-0.17514 	0.28728
    -0.48485 	-2.19967 	3.11798 	-0.34289 	0.00000 	-3.49274 	0.49944 	-0.34627 	-0.79726 	0.40419 	0.00000 	0.25019 	0.00000
    -0.38081 	-0.80504 	1.42256 	-0.18440 	0.00000 	-1.60841 	0.51263 	-0.33287 	-0.04213 	0.00000 	0.34189 	0.14923 	0.00000
    -0.35774 	-1.29695 	2.11457 	-0.29809 	0.00000 	-2.21604 	0.56709 	-0.34741 	-0.33964 	0.17556 	0.09301 	0.13036 	0.00000
    -0.54929 	-1.00403 	2.00552 	-0.29117 	0.00000 	-2.11507 	0.47503 	-0.44130 	0.00000 	-0.18147 	0.67147 	0.05455 	0.24920
    -0.47724 	-1.33588 	2.40366 	-0.35836 	0.00000 	-2.39386 	0.44171 	-0.44545 	-0.49459 	0.25080 	0.80050 	0.00000 	0.22973
    ];

%% relate the coefficients to the selected model
switch typeModel
    case 'PGA,PGV'
        idModel = 1;
    case 'PGA,Tm'
        idModel = 2;
    case 'PGA,IA'
        idModel = 3;
    case 'PGA,SI'
        idModel = 4;
        X_min = [-4.60507 	-2.97874 	-4.60517];
        X_max = [0.90361 	6.54505 	-0.22314];
        X_Input = [x_lnIM1, x_lnIM2, x_lnKy];
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        weightMatrix = [
            24.68510 	37.38324 	0.19721 	-2.64594 	-0.26632 	1.61250 	14.03180 	1.74134 	7.27808 	-1.88019 	4.42061 	2.49491
            -1.47447 	2.29020 	1.73999 	3.42215 	3.86346 	1.56745 	-9.72195 	1.00841 	1.35401 	4.11356 	0.29778 	-1.65735
            -19.41897 	-29.64456 	-0.66860 	0.40497 	-0.90703 	-1.54834 	1.13162 	0.01233 	-5.49525 	1.14239 	-4.28819 	-1.21459
            ];
        weightVector = [
            12.96116 	0.84559 	5.39416 	16.46046 	1.56076 	2.41704 	0.17863 	0.38301 	1.19295 	0.68425 	2.54089 	17.83211
            ];
        biasVector = [
            6.86876 	5.92318 	-3.35440 	-3.78817 	-2.70990 	-0.62144 	-0.03527 	0.15635 	0.72718 	1.67531 	1.40297 	2.49474
            ];
        biasScalar = -12.94178;
    case 'PGA,Tm,IA'
        idModel = 5;
        X_min = [-4.60507 	-2.84107 	-8.94085 	-4.60517];
        X_max = [0.90361 	0.94114 	3.26350 	-0.22314];
        X_Input = [x_lnIM1, x_lnIM2, x_lnIM3, x_lnKy];
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        weightMatrix = [
            37.93383 	2.12278 	19.99343 	101.42907 	2.38172 	3.98866 	95.02582 	2.88263 	-0.15519 	-0.43311 	-9.14917 	0.09220 	0.69903 	-5.42756 	-2.64335 	-2.38003
            -3.56045 	-0.20769 	-0.06585 	1.07624 	2.07759 	0.24208 	1.02540 	2.45117 	0.88774 	0.81605 	-1.49389 	0.42739 	0.16925 	0.76548 	0.29566 	-0.01038
            -4.18603 	-1.52708 	0.47328 	-0.50034 	-10.91032 	-0.66157 	-0.41150 	-11.96431 	1.76804 	1.62843 	10.22408 	1.26543 	0.63316 	10.98628 	3.54180 	1.68797
            -29.32156 	-1.36384 	-16.18281 	-79.60438 	-1.78012 	-2.65039 	-74.58175 	-1.83960 	-0.39400 	-0.44269 	-0.49654 	-0.51625 	-0.51154 	-1.51155 	0.94091 	-0.13549
            ];
        weightVector = [
            0.55327 	10.93919 	17.32187 	24.01827 	2.32998 	25.97553 	-27.86036 	-2.31487 	2.44955 	10.69566 	5.14484 	-21.60204 	17.46231 	0.23520 	22.64736 	2.47684
            ];
        biasVector = [
            10.81106 	2.05661 	5.09482 	20.10691 	6.60636 	2.17342 	18.96825 	7.10448 	-1.44379 	0.88369 	2.37070 	0.83738 	0.66230 	-2.81570 	-3.95023 	-1.77071
            ];
        biasScalar = -32.28443;
    case 'PGA,Tm,CAV'
        idModel = 6;
        X_min = [-4.60507 	-2.84107 	-5.03974 	-4.60517];
        X_max = [0.90361 	0.94114 	1.68690 	-0.22314];
        X_Input = [x_lnIM1, x_lnIM2, x_lnIM3, x_lnKy];
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        weightMatrix = [
            0.51472 	1.54351 	-18.33361 	-1.91100 	18.32585 	1.14639 	1.56600 	33.52092 	5.39931 	-0.37561 	2.77294 	6.75228
            0.08656 	1.71106 	-2.52011 	-1.72543 	-0.48177 	0.95274 	-0.92905 	1.28180 	0.70664 	2.36309 	-0.00998 	-5.59554
            0.91265 	0.05237 	10.75115 	0.31505 	0.06468 	-0.64903 	-0.54612 	-0.29392 	-0.15449 	0.77398 	-0.46196 	-23.67286
            -0.57593 	-1.28518 	-3.30232 	1.30824 	-15.04335 	-0.06590 	-5.41203 	-25.14656 	-3.23960 	0.23086 	-2.18532 	-0.89113
            ];
        weightVector = [
            4.25413 	6.51284 	0.31632 	6.51733 	15.92102 	2.35746 	-12.18851 	0.77172 	1.26035 	0.77372 	9.86389 	7.26498
            ];
        biasVector = [
            -0.52012 	-2.14262 	6.72820 	2.08684 	5.09567 	-0.46013 	6.68137 	5.89462 	0.92868 	1.81529 	1.43742 	23.59869
            ];
        biasScalar = -20.97565;
    case 'PGA,PGV,IA'
        idModel = 7;
        X_min = [-4.60507 	-2.02049 	-8.94085 	-4.60517];
        X_max = [0.90361 	5.57303 	3.26350 	-0.22314];
        X_Input = [x_lnIM1, x_lnIM2, x_lnIM3, x_lnKy];
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        weightMatrix = [
            19.66794 	-1.02859 	-0.32034 	4.30401 	6.87869 	0.33128 	3.13846 	0.77366 	7.60347 	4.08620 	0.85884 	13.32424 	2.55366 	-3.46158
            -0.96470 	0.55760 	1.18035 	-0.86394 	1.19777 	3.89074 	1.77034 	2.35282 	1.15050 	0.16011 	5.41583 	1.60001 	-1.02419 	2.54643
            0.23414 	2.43597 	0.33686 	-3.02577 	-0.49655 	0.69169 	-0.74367 	0.89264 	-0.59715 	-1.07300 	-4.14860 	-0.77708 	-0.30182 	3.37812
            -15.64298 	-0.67283 	-0.20426 	-0.10365 	-5.29327 	-1.12591 	-2.56621 	-1.00556 	-5.76940 	-3.08174 	-1.38436 	-10.31695 	-0.99106 	-0.20014
            ];
        weightVector = [
            11.75592 	2.10988 	4.12401 	-2.47051 	-5.40175 	-0.98675 	3.17762 	1.46834 	-6.14209 	2.81432 	0.32317 	3.80069 	8.91106 	1.02023
            ];
        biasVector = [
            5.18066 	-1.07671 	-1.37086 	2.93083 	2.39075 	0.79993 	1.09698 	0.50526 	2.61911 	1.23251 	0.20118 	2.81611 	1.60513 	0.96340
            ];
        biasScalar = -12.62705;
    case 'PGA,PGV,CAV'
        idModel = 8;
        X_min = [-4.60507 	-2.02049 	-5.03974 	-4.60517];
        X_max = [0.90361 	5.57303 	1.68690 	-0.22314];
        X_Input = [x_lnIM1, x_lnIM2, x_lnIM3, x_lnKy];
        % modify data outside the training ranges
        for i = 1:length(X_min)
            ans = (X_Input(:,i)<X_min(i));
            X_Input(ans,i) = X_min(i);
            ans = (X_Input(:,i)>X_max(i));
            X_Input(ans,i) = X_max(i);
        end
        weightMatrix = [
            23.09030 	-5.51855 	2.56618 	-0.86653 	-3.16278 	2.80479 	11.90172 	2.86222 	26.75168 	17.52622 	10.31932 	0.55725 	1.21368 	2.09261
            -0.83263 	6.12298 	-0.97112 	0.44282 	3.38490 	-1.66246 	0.73233 	-3.75039 	1.23060 	0.93109 	0.48110 	4.15462 	0.10624 	-0.15052
            0.10343 	4.49678 	0.31555 	0.83871 	0.43644 	1.01282 	-0.42383 	1.01357 	-0.58279 	-0.35260 	-0.32681 	2.24750 	-0.92866 	-0.25652
            -18.42917 	-0.11543 	-2.05311 	-0.09517 	-0.05441 	-1.66812 	-9.62997 	-0.18714 	-21.26291 	-13.83342 	-8.35947 	0.30686 	-0.09752 	-1.19321
            ];
        weightVector = [
            14.10079 	0.25992 	4.95496 	9.86759 	5.42251 	-3.37885 	4.22978 	5.14877 	8.32845 	-17.22731 	-11.00935 	0.12378 	5.31952 	11.77609
            ];
        biasVector = [
            6.19227 	-3.52275 	1.14701 	0.11424 	-2.90170 	0.54921 	1.92414 	2.27593 	5.33844 	4.32269 	2.53319 	0.16553 	-0.25505 	0.92050
            ];
        biasScalar = -10.59041;
end

%% predict the sliding displacement
% rigid sliding displacement
if idModel>=4
    X_norm = (1-(-1))*(X_Input-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]))-1;
    lnDr = (2./(1+exp(-2*(X_norm*weightMatrix+repmat(biasVector,[n_data,1]))))-1)*weightVector'+biasScalar;
else
    X_Input = [ones(size(x_lnIM1)), exp(x_lnKy)./exp(x_lnIM1), (exp(x_lnKy)./exp(x_lnIM1)).^2,...
        (exp(x_lnKy)./exp(x_lnIM1)).^3, (exp(x_lnKy)./exp(x_lnIM1)).^4, x_lnIM1, x_lnIM2];
    lnDr = (coeff_D_SR08(idModel,:)*X_Input')';
end

% modification term
input_CorrectModel = [ones(size(x_Ts)), x_Ts, x_Ts.^2, x_Ts.^3, log(x_Ts), x_Ts.*log(x_Ts),...
    x_h_ratio, x_h_ratio.^2, x_IR, x_IR.^2, x_Ts.*x_h_ratio, x_Ts.*x_IR, x_IR.*x_h_ratio];
Mf = (e_Mf(idModel,:)*input_CorrectModel')';

% final displacement
median_D = exp(lnDr+Mf);
median_D(x_Ts<0.02) = median_D(x_Ts<0.02);
median_D = reshape(median_D,n_row,n_col);

% standard deviation
ratio_KyKmax = exp(X_Input(:,end))./exp(X_Input(:,1));
range_KyKmax = [0.05,0.95];
ratio_KyKmax(ratio_KyKmax<range_KyKmax(1)) = range_KyKmax(1);
ratio_KyKmax(ratio_KyKmax>range_KyKmax(2)) = range_KyKmax(2);
input_SigmaRigid = [ones(size(x_lnKy)), ratio_KyKmax];
input_SigmaFlexi = [ones(size(x_lnKy)), ratio_KyKmax, ratio_KyKmax.^2];
sigmaFlexi_lnD = (coeff_SigmaFlexi(idModel,:)*input_SigmaFlexi')';
sigmaRigid_lnD = (coeff_SigmaRigid(idModel,:)*input_SigmaRigid')';
sigma_lnD = sigmaFlexi_lnD;
sigma_lnD(x_Ts<0.02) = sigmaRigid_lnD(x_Ts<0.02);
sigma_lnD = reshape(sigma_lnD,n_row,n_col);
