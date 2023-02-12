
function [epoch_acc, epoch_svd, epoch_best_Z, epoch_best_acc, svd_at_best_epoch, acc_at_best_epoch, ...
    epoch_accuracy_by_each_bias, each_bias_acc_at_best_epoch, ...
    epoch_partial_acc, partial_acc_at_best_epoch] = ...
    run_iteration(X, Z, k, iter, slack_ratio, Z_num, partial_supervised, partial_n, biases_num, biases_dim, include_complex_Z, complex_index)

    %% X and Z are n x d1 and n x d2 matrices of the same number of rowss
    %% Zid is of length n x 1 and it gives an "identifier" to each Z (for example,
    %% if Z is a one-hot vector over multiclasss [K], then Zid would give a number between
    %% 1 and K, identifying the class - to cluster the Zs).
    %% The reason we need Zid is that in principle we can create a permutation of [n]
    %% and many of them would be "good" -- if we just swap Zs that have the same class.
    %% The Zid helps identify which Zs arae the same when calculating the accuracy of
    %% the permutation.
    %% k is the number of singular values used.
    %% iter is the number of iterations to run the algorithm.
    
    
    n_x = size(X, 1);
    n_z = size(X, 1);
    origZ = Z;
    
    %% start by permuting Z randomly
    rand_pi = randperm(size(Z,1));
    Z = Z(rand_pi, :);
    new_pi = rand_pi;

    epoch_best_acc = 0;
    svd_at_best_epoch = 0;
    acc_at_best_epoch = 0;
    partial_acc_at_best_epoch = 0;
    epoch_acc = [];
    epoch_partial_acc = [];
    epoch_svd = [];

    epoch_accuracy_by_each_bias = [];

    for i=1:iter
        if i ~= 1
            if include_complex_Z == true
                Z = find_assignment_complex(X, Z, U, V, slack_ratio, Z_num );
            else
                Z = find_assignment(X, Z, U, V, slack_ratio, Z_num );
            end
        end
        
%         new_pi = new_pi(pi);
%         Z = origZ(new_pi, :);

        C = X' * Z;
        [U,S,V] = svds(C, k);
        s = sum(diag(S));
    
        
        score = sum(all(Z(1:n_x, :) == origZ(1:n_x, :), 2));
        iter_score = score / n_x;
        
        biases_by_type = [];
        for i = 1:biases_num
            if any(i==complex_index)
                a = Z(:, biases_dim(i)+1: biases_dim(i+1));
                b = origZ(:, biases_dim(i)+1: biases_dim(i+1));
                acc = sum(1 - diag(pdist2(a, b, 'cosine'))) / n_x;
            else
                acc = sum(all(Z(:, biases_dim(i)+1: biases_dim(i+1)) == origZ(:, biases_dim(i)+1: biases_dim(i+1)), 2)) / n_x;
            end
            biases_by_type = [biases_by_type acc];
        end



        if partial_supervised == true

            score_partial_n = sum(all(Z(1:partial_n, :) == origZ(1:partial_n, :), 2));
            score_partial_n = score_partial_n / partial_n;
            epoch_partial_acc = [epoch_partial_acc; score_partial_n];

            if score_partial_n >= partial_acc_at_best_epoch
                svd_at_best_epoch = s;
                acc_at_best_epoch = iter_score;
                each_bias_acc_at_best_epoch = biases_by_type;
                
                epoch_best_Z = Z;
                partial_acc_at_best_epoch = score_partial_n;
            end
        else
            if s > svd_at_best_epoch
                svd_at_best_epoch = s;
                acc_at_best_epoch = iter_score;
                each_bias_acc_at_best_epoch = biases_by_type;

                epoch_best_Z = Z;
            end
        end


        
        if iter_score > epoch_best_acc
            epoch_best_acc = iter_score;
        end

    
        epoch_acc = [epoch_acc; score / n_x];
        epoch_svd = [epoch_svd; s];
        epoch_accuracy_by_each_bias = [epoch_accuracy_by_each_bias; biases_by_type];

    end

end

