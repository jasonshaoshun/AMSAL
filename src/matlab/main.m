
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



tic;

load(fullfile(relative_path_to_project, dataset_path))

x_train = double(x_train);
y_p_train = double(y_p_train);
y_p_train = reshape(y_p_train, size(x_train, 1), []);

if ~exist('y_m_train')
    disp("y_m_train not exists replaced by 0");
    y_m_train = zeros(size(x_train, 1), 1);
else
    y_m_train = reshape(y_m_train, size(x_train, 1), []);
    y_m_train = double(y_m_train);
end

% if experiments == "political" && bias_name == "gender"
%     disp("protected attribute is not binary, clean the dataset ...");
%     order = y_p_train == 0 | y_p_train == 1;
%     x_train = x_train(order, :);
%     y_p_train = y_p_train(order, :);
%     y_m_train = y_m_train(order, :);
% else
%     disp("do not clean the dataset");
% end

if N_max > size(x_train, 1)
    N_max = size(x_train, 1);
end

x_train = x_train(1:N_max, :);
y_m_train = y_m_train(1:N_max, :);
y_p_train = y_p_train(1:N_max, :);

partial_n = floor(N_max * partial_ratio);

best_iter_score = 0;
best_full_score = 0;
best_each_bias_score = 0;

kmeans_best_iter_svd = 0;
kmeans_best_iter_score = 0;
best_iter_svd = 0;

Slack10_Accuracy_epoch = [];
Slack10_SVD_epoch = [];
Slack20_Accuracy_epoch = [];
Slack20_SVD_epoch = [];
Slack30_Accuracy_epoch = [];
Slack30_SVD_epoch = [];

Slack10_partial_Accuracy_epoch = [];
Slack20_partial_Accuracy_epoch = [];
Slack30_partial_Accuracy_epoch = [];

Slack10_Accuracy_by_each_biases_epoch = [];
Slack20_Accuracy_by_each_biases_epoch = [];
Slack30_Accuracy_by_each_biases_epoch = [];

Kmeans_Accuracy_epoch = [];
Kmeans_SVD_epoch = [];

rng(0);

if save_partial_n == true
    partial_n_in_file_name = strcat("_np-", string(partial_ratio));
else
    partial_n_in_file_name = "";
end

% if save_seed == true
%     seed_in_file_name = strcat("_sd-", seed);
% else
%     seed_in_file_name = "";
% end

if save_assignment_method == true
    assignment_method = "SAL_";
else
    assignment_method = "";
end

for i =1:epoch_num

    best_full_acc = 0;
    best_each_bias_acc = 0;
    best_svd = 0;
    best_Z = 0;
    best_partial_acc = 0;
    best_slack = '';
    
    [X, Z, Y] = shuffle_data(x_train, y_p_train, y_m_train);
    N = size(X, 1);

    [epoch_acc, epoch_svd, epoch_best_Z, epoch_best_acc, svd_at_best_epoch, acc_at_best_epoch, ...
        epoch_accuracy_by_each_bias, each_bias_acc_at_best_epoch, ...
    epoch_partial_acc, partial_acc_at_best_epoch] = ...
    run_by_slack(X, Z, k, iter, 0.1 , partial_supervision, partial_n, biases_num, biases_dim, include_complex_Z, complex_index);
    
    [Slack10_Accuracy_epoch, Slack10_SVD_epoch, Slack10_Accuracy_by_each_biases_epoch, ...
        best_full_acc, best_each_bias_acc, best_slack, best_svd, best_Z, ...
    Slack10_partial_Accuracy_epoch, best_partial_acc] = ...
    update_slack_scores(Slack10_Accuracy_epoch, Slack10_SVD_epoch, Slack10_Accuracy_by_each_biases_epoch, ...
    epoch_acc, epoch_svd, epoch_accuracy_by_each_bias, ...
    best_full_acc, best_each_bias_acc, best_svd, best_Z, ...
    acc_at_best_epoch, each_bias_acc_at_best_epoch, svd_at_best_epoch, epoch_best_Z, best_slack, partial_supervision, 0.1,...
    Slack10_partial_Accuracy_epoch, epoch_partial_acc, partial_acc_at_best_epoch, best_partial_acc);



    [epoch_acc, epoch_svd, epoch_best_Z, epoch_best_acc, svd_at_best_epoch, acc_at_best_epoch, ...
        epoch_accuracy_by_each_bias, each_bias_acc_at_best_epoch, ...
    epoch_partial_acc, partial_acc_at_best_epoch] = ...
    run_by_slack(X, Z, k, iter, 0.2 , partial_supervision, partial_n, biases_num, biases_dim, include_complex_Z, complex_index);
    
    [Slack20_Accuracy_epoch, Slack20_SVD_epoch, Slack20_Accuracy_by_each_biases_epoch, ...
        best_full_acc, best_each_bias_acc, best_slack, best_svd, best_Z, ...
    Slack20_partial_Accuracy_epoch, best_partial_acc] = ...
    update_slack_scores(Slack20_Accuracy_epoch, Slack20_SVD_epoch, Slack20_Accuracy_by_each_biases_epoch, ...
    epoch_acc, epoch_svd, epoch_accuracy_by_each_bias, ...
    best_full_acc, best_each_bias_acc, best_svd, best_Z, ...
    acc_at_best_epoch, each_bias_acc_at_best_epoch, svd_at_best_epoch, epoch_best_Z, best_slack, partial_supervision, 0.2,...
    Slack20_partial_Accuracy_epoch, epoch_partial_acc, partial_acc_at_best_epoch, best_partial_acc);



    [epoch_acc, epoch_svd, epoch_best_Z, epoch_best_acc, svd_at_best_epoch, acc_at_best_epoch, ...
        epoch_accuracy_by_each_bias, each_bias_acc_at_best_epoch, ...
    epoch_partial_acc, partial_acc_at_best_epoch] = ...
    run_by_slack(X, Z, k, iter, 0.3 , partial_supervision, partial_n, biases_num, biases_dim, include_complex_Z, complex_index);
    
    [Slack30_Accuracy_epoch, Slack30_SVD_epoch, Slack30_Accuracy_by_each_biases_epoch, ...
        best_full_acc, best_each_bias_acc, best_slack, best_svd, best_Z, ...
    Slack30_partial_Accuracy_epoch, best_partial_acc] = ...
    update_slack_scores(Slack30_Accuracy_epoch, Slack30_SVD_epoch, Slack30_Accuracy_by_each_biases_epoch, ...
    epoch_acc, epoch_svd, epoch_accuracy_by_each_bias, ...
    best_full_acc, best_each_bias_acc, best_svd, best_Z, ...
    acc_at_best_epoch, each_bias_acc_at_best_epoch, svd_at_best_epoch, epoch_best_Z, best_slack, partial_supervision, 0.3,...
    Slack30_partial_Accuracy_epoch, epoch_partial_acc, partial_acc_at_best_epoch, best_partial_acc);


    fprintf("best accuracy on whole dataset %.2f\n", best_full_acc)
%     fprintf("best_partial_acc %.2f\n", best_partial_acc)
    
    if (partial_supervision == false) && (partial_ratio == 0)
%         If run in unsupervised manner, update only if
        if svd_at_best_epoch > best_iter_svd

            best_full_score = best_full_acc;
            best_each_bias_score = best_each_bias_acc;
            best_iter_svd = best_svd;
            best_epoch_slack = best_slack;
    
            best_iter_Z = best_Z;
            best_iter_X = X;
            best_iter_Y = Y;
            best_iter_original_Z = Z;
        end
    else
        if best_partial_acc > best_iter_score
    
            best_iter_score = best_partial_acc;
            best_full_score = best_full_acc;
            best_each_bias_score = best_each_bias_acc;
            best_iter_svd = best_svd;
            best_epoch_slack = best_slack;
    
            best_iter_Z = best_Z;
            best_iter_X = X;
            best_iter_Y = Y;
            best_iter_original_Z = Z;
        end
    end


    if  run_kmeans == true
        [Z_unq, ia_Z, ib_Z ] = unique(Z, 'rows');
        num_classes = length(Z_unq);
    
        occurences_Z = zeros(num_classes, 1);
        for i = 1:num_classes
            occurences_Z(i, 1) = sum(all(Z_unq(i, :) == Z, 2));
        end
    
        [B_Z I_Z] = sort(occurences_Z);
        
        
        idx = kmeans(X, num_classes);
        [idx_unq, ia_idx, ib_index] = unique(idx, 'rows');
    
        occurences_idx = zeros(num_classes, 1);
        for i = 1:num_classes
            occurences_idx(i, 1) = sum(all(idx_unq(i, :) == idx, 2));
        end
    
        [B_idx I_idx] = sort(occurences_idx);
    
    
        kmeans_Z = zeros(N, size(Z, 2));
        for i = 1:num_classes
            count_Z_i = sum(idx == I_idx(i));
            kmeans_Z(idx == I_idx(i), :) = repmat(Z_unq(I_Z(i), :), count_Z_i, 1);
        end
    
        kmeans_epoch_score = sum(all(kmeans_Z == Z, 2))/N;
    
    
    
        C = X' * kmeans_Z;
        [U,S,V] = svds(C, k);
        kmeans_epoch_svd = sum(diag(S));
        Kmeans_Accuracy_epoch = [Kmeans_Accuracy_epoch kmeans_epoch_score];
        Kmeans_SVD_epoch = [Kmeans_SVD_epoch kmeans_epoch_svd];
    
    %     fprintf('Epoch ACC at %.3f\n', kmeans_epoch_svd_score/N);
        if kmeans_epoch_svd > kmeans_best_iter_svd
            kmeans_best_iter_svd = kmeans_epoch_svd;
            kmeans_best_iter_score = kmeans_epoch_score;
            kmeans_best_iter_Z = kmeans_Z;
            kmeans_best_iter_X = X;
            kmeans_best_iter_Y = Y;
            kmeans_best_iter_original_Z = Z;
        end
    end
    toc;
end


if experiments == "biography"
    experiments_in_short = "bio";
elseif experiments == "deepmoji"
    experiments_in_short = "dp";
elseif experiments == "word-embedding"
    experiments_in_short = "wd";
elseif experiments == "political"
    experiments_in_short = "pl";
elseif experiments == "twitter-short-sentiment"
    experiments_in_short = "tss";
elseif experiments == "twitter-concat-sentiment"
    experiments_in_short = "tcs";
elseif experiments == "twitter-short-sentiment-v4"
    experiments_in_short = "tssv4";
elseif experiments == "twitter-concat-sentiment-v4"
    experiments_in_short = "tcsv4";
elseif experiments == "twitter-different-partial-n-short"
    experiments_in_short = 'tssDP';
elseif experiments == "twitter-different-partial-n-concat"
    experiments_in_short = 'tcsDP';
elseif experiments == "twitter-different-partial-n-short-v4"
    experiments_in_short = 'tssDPv4';
elseif experiments == "twitter-different-partial-n-concat-v4"
    experiments_in_short = 'tcsDPv4';
elseif experiments == "ratio_on_race"
    experiments_in_short = 'r_r';
elseif experiments == "ratio_on_sent"
    experiments_in_short = 'r_s';
else
    experiments_in_short = experiments;
end


if model_name == "BertModel"
    model_name_in_short = "BT";
elseif model_name == "FastText"
    model_name_in_short = "FT";
elseif model_name == "05"
    model_name_in_short = "05";
elseif model_name == "glove"
    model_name_in_short = "gl";
elseif model_name == "AlbertModel"
    model_name_in_short = "Albt";
elseif model_name == "RobertaModel"
    model_name_in_short = "Robt";
elseif model_name == "GPT2Model"
    model_name_in_short = "GPT";
elseif model_name == "Deepmoji"
    model_name_in_short = "DM";
else
    model_name_in_short = model_name;
end

if bias_name == "age_gender"
    bias_name_in_short = "AG";
else
    bias_name_in_short = bias_name;
end

sheet_name = strcat(experiments_in_short, '_', model_name_in_short, partial_n_in_file_name, '_sup-', string(partial_supervision));
disp("sheet_name");
disp(sheet_name);
filename = fullfile(relative_path_to_project, 'data/epoch.xlsx');

if (partial_supervision == false) && (partial_ratio == 0)
    Slack10_partial_Accuracy_epoch = zeros(iter, epoch_num);
    Slack20_partial_Accuracy_epoch = zeros(iter, epoch_num);
    Slack30_partial_Accuracy_epoch = zeros(iter, epoch_num);
end

T = table(Slack10_Accuracy_epoch, Slack10_partial_Accuracy_epoch, Slack10_SVD_epoch, Slack10_Accuracy_by_each_biases_epoch);
writetable(T,filename,'Sheet', sheet_name,'Range','A1')

T = table(Slack20_Accuracy_epoch, Slack20_partial_Accuracy_epoch, Slack20_SVD_epoch, Slack20_Accuracy_by_each_biases_epoch);
writetable(T,filename,'Sheet', sheet_name,'Range','A200')

T = table(Slack30_Accuracy_epoch, Slack30_partial_Accuracy_epoch, Slack30_SVD_epoch, Slack30_Accuracy_by_each_biases_epoch);
writetable(T,filename,'Sheet', sheet_name,'Range','A400')

if run_kmeans == true
    T = table(Kmeans_Accuracy_epoch, Kmeans_SVD_epoch);
    writetable(T, filename, 'Sheet', sheet_name, 'Range', 'A600');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C = best_iter_X' * best_iter_Z;
% C = best_iter_X' * best_iter_original_Z;
[U_experiment, S, V] = svd(C);

C = best_iter_X' * best_iter_original_Z;
[U_best, S, V] = svd(C);

assignment_accuracy = (sum(all(best_iter_Z == best_iter_original_Z, 2)) / N_max) * 100;

fprintf('\n\n\n\n\n\n\n\n\n\n\n\n\n==============================================================================================================================================================================\n')
fprintf('End, Model: %s, Bias: %s, Best Slack %s per cent, dataset size %d, dataset %s.\n', model_name, bias_name, best_epoch_slack, N, dataset_path);
fprintf('Best SVD %.3f, best_iter_score %.3f\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n', best_iter_svd, best_iter_score);
if partial_ratio == 0
    save(fullfile(save_path, strcat(assignment_method, bias_name, '_Sal', partial_n_in_file_name, '.mat')), 'best_epoch_slack', 'U_experiment', 'U_best', 'best_iter_svd', 'best_iter_Z', 'best_iter_X', 'best_iter_Y', 'best_iter_original_Z', 'assignment_accuracy');
else
    save(fullfile(save_path, strcat(assignment_method, bias_name, '_partialSup', partial_n_in_file_name, '.mat')), 'best_epoch_slack', 'U_experiment', 'U_best', 'best_iter_svd', 'best_iter_Z', 'best_iter_X', 'best_iter_Y', 'best_iter_original_Z', 'assignment_accuracy');
end


if save_oracle == true
    best_iter_Z = best_iter_original_Z;
    U_experiment = U_best;
    assignment_accuracy = 100.00;
    save(fullfile(save_path, strcat(assignment_method, bias_name, '_Oracle', partial_n_in_file_name, '.mat')), 'U_experiment', 'best_iter_X', 'best_iter_Y', 'best_iter_Z', 'assignment_accuracy');
end

% Old version
% C = kmeans_best_iter_X' * kmeans_best_iter_Z;
% [U_experiment, S, V] = svd(C);
% 
% C = kmeans_best_iter_X' * kmeans_best_iter_original_Z;
% [U_best, S, V] = svd(C);

if run_kmeans == true
    C = kmeans_best_iter_X' * kmeans_best_iter_Z;
    [U_experiment, S, V] = svd(C);
    
    C = kmeans_best_iter_X' * kmeans_best_iter_original_Z;
    [U_best, S, V] = svd(C);
    
    best_iter_X = kmeans_best_iter_X;
    best_iter_Y = kmeans_best_iter_Y;
    best_iter_Z = kmeans_best_iter_Z;
    best_iter_original_Z = kmeans_best_iter_original_Z;
    assignment_accuracy = (sum(all(kmeans_best_iter_Z == kmeans_best_iter_original_Z, 2)) / N_max) * 100;
    fprintf('\n\n\n\n\n\n\n\n\n\n\n\n\n==============================================================================================================================================================================\n')
    fprintf('End, Model: %s, Bias: %s, dataset size %d, dataset %s.\n', model_name, bias_name, N, dataset_path);
    fprintf('KMEANS Best SVD %.3f, best_iter_score %.3f\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n', kmeans_best_iter_svd, kmeans_best_iter_score);
    save(fullfile(save_path, strcat(assignment_method, bias_name, '_Kmeans.mat')), 'U_experiment', 'U_best', 'kmeans_best_iter_svd', 'best_iter_Z', 'best_iter_X', 'best_iter_Y', 'best_iter_original_Z', 'assignment_accuracy');
end


if run_kmeans == false
    kmeans_best_iter_svd = 0;
    kmeans_best_iter_score = 0;
end

if ~isfile('data/full.xlsx')
    writetable(table({'experiments'}, {'Language Model Name'}, {'Bias Name'}, {'Bias Name In Short'}, {'Number of Samples'}, ...
        {'Ratio Of Supervised Samples'}, {'Best Sum Of Singular Values'}, {'Accuracy At Best Epoch'}, ...
        {'Partial Accuracy at Best Epoch'}, {'Slack ratio'}, {'KMeans Best Sum Of Singular Values'}, ...
        {'KMeans Accuracy At Best SVD'}, {'each bias accuracy at best epoch'}), ...
        fullfile(relative_path_to_project, 'data/full.xlsx'),'WriteRowNames', false);
end

% For matlab19 later
T = table({experiments}, {model_name}, {bias_name}, {bias_name_in_short}, N, {partial_ratio}, best_iter_svd, best_full_score, best_iter_score, {best_epoch_slack}, kmeans_best_iter_svd, kmeans_best_iter_score, best_each_bias_score);
writetable(T, fullfile(relative_path_to_project, 'data/full.xlsx'), 'WriteMode', 'Append', 'WriteVariableNames', false);



% % For matlab18
% T = table({model_name}, {bias_name}, N, best_iter_svd, best_full_score, best_each_bias_score, best_iter_score, {best_epoch_slack}, kmeans_best_iter_svd, kmeans_best_iter_score);
% writetable(T, 'data/temp.xlsx', 'WriteVariableNames', false);
% 
% old_full_result_table = readtable('data/political_full_result.xlsx', 'ReadVariableNames', false);
% new_full_result_table = readtable('data/temp.xlsx', 'ReadVariableNames', false);
% T = [old_full_result_table; new_full_result_table];
% writetable(T, 'data/political_full_result.xlsx', 'WriteVariableNames', false);

toc;




function [Slack_Accuracy_epoch, Slack_SVD_epoch, Slack_Accuracy_by_each_biases_epoch, ...
    best_full_acc, best_each_bias_acc, best_slack, best_svd, best_Z, ...
    Slack_partial_Accuracy_epoch, best_partial_acc] = ...
    update_slack_scores(Slack_Accuracy_epoch, Slack_SVD_epoch, Slack_Accuracy_by_each_biases_epoch, ...
    epoch_acc, epoch_svd, epoch_accuracy_by_each_bias, ...
    best_full_acc, best_each_bias_acc, best_svd, best_Z,...
    acc_at_best_epoch, each_bias_acc_at_best_epoch, svd_at_best_epoch, epoch_Z, best_slack, partial_supervision, slack_ratio, ...
    Slack_partial_Accuracy_epoch, epoch_partial_acc, partial_acc_at_best_epoch, best_partial_acc)

    Slack_Accuracy_epoch = [Slack_Accuracy_epoch epoch_acc];
    Slack_SVD_epoch = [Slack_SVD_epoch epoch_svd];
    Slack_Accuracy_by_each_biases_epoch = [Slack_Accuracy_by_each_biases_epoch epoch_accuracy_by_each_bias];

    if partial_supervision == true
        Slack_partial_Accuracy_epoch = [Slack_partial_Accuracy_epoch epoch_partial_acc];
        if partial_acc_at_best_epoch > best_partial_acc
            if slack_ratio == 0.1
                best_slack = '10 % slack';
            elseif slack_ratio == 0.2
                best_slack = '20 % slack';
            elseif slack_ratio == 0.3
                best_slack = '30 % slack';
            else
                best_slack = '? % slack';
            end

            best_partial_acc = partial_acc_at_best_epoch;
            best_full_acc = acc_at_best_epoch;
            best_each_bias_acc = each_bias_acc_at_best_epoch;
            best_svd = svd_at_best_epoch;
            best_Z = epoch_Z;

        end
    else
        if svd_at_best_epoch > best_svd
            if slack_ratio == 0.1
                best_slack = '10 % slack';
            elseif slack_ratio == 0.2
                best_slack = '20 % slack';
            elseif slack_ratio == 0.3
                best_slack = '30 % slack';
            else
                best_slack = '? % slack';
            end
            
            best_full_acc = acc_at_best_epoch;
            best_each_bias_acc = each_bias_acc_at_best_epoch;
            best_svd = svd_at_best_epoch;
            best_Z = epoch_Z;
            
        end

    end

end



function [epoch_acc, epoch_svd, epoch_best_Z, epoch_best_acc, svd_at_best_epoch, acc_at_best_svd, ...
    epoch_accuracy_by_each_bias, each_bias_acc_at_best_epoch, ...
    epoch_partial_acc, partial_acc_at_best_svd] = ...
    run_by_slack(X, Zid, k, iter, slack_ratio, partial_supervision, partial_n, biases_num, biases_dim, include_complex_Z, complex_index)
    
    if include_complex_Z == true
        Z_distribution = [];
    else
        [Z_unq, unq_I, unq_J] = unique(Zid, 'rows');
    
        valueSet = zeros(length(Z_unq), 1);
        for i = 1:length(Z_unq)
            valueSet(i, 1) = sum(all(Z_unq(i, :) == Zid, 2));
        end
    
        keySet = {};
        for i = 1:length(Z_unq)
            keySet{end+1} = num2str(Z_unq(i, :));
        end
        
        Z_distribution = containers.Map(keySet,valueSet);
    end

    [epoch_acc, epoch_svd, epoch_best_Z, epoch_best_acc, svd_at_best_epoch, acc_at_best_svd,...
        epoch_accuracy_by_each_bias, each_bias_acc_at_best_epoch, ...
        epoch_partial_acc, partial_acc_at_best_svd] = ...
        run_iteration(X, Zid, k, iter, slack_ratio, Z_distribution, partial_supervision, partial_n, biases_num, biases_dim, include_complex_Z, complex_index);

end




function [X, Zid, Y] = shuffle_data(x_train, y_p_train, y_m_train)

%     rand_pi = randperm(size(x_train,1));
    rand_pi = 1:size(x_train,1);
    X = x_train(rand_pi, :);
    Zid = y_p_train(rand_pi, :);
    Y = y_m_train(rand_pi, :);

end


