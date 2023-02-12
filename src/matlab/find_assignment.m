function [new_Z, E, T] = find_assignment(X, Z, U, V, slack_ratio, Z_distribution)

    %% U needs to be dim_of_X x d and V needs to be dim_of_Z x d
    %% d is the number of singular values we take
    
    %% min_how_many_Xs_each_Z_receives and max_how_many_Xs_each_Z_receives
    %% are 1 * n_z matrices such that C[1,j] = integer, and that's the min or
    %% the max that the jth z can receive.
    
    %% need to decide on f, intcon, A, b
    
    %% note that the size of A in the slides is n_x*n_z -- hence the number of integer 0/1 variables for
    %% the ILP we have is n_x * n_z - each variable is 0 or 1 and tells whether there is an edge
    %% in the match between X_i and Z_j.
    
    [Z_unq, unq_I, unq_J] = unique(Z, 'rows');
    
    %% X is n_x * dim_of_x_representation, Z is n_z * dim_of_z_representation
    n_x = size(X, 1);
    n_z = length(unq_I);
    n_d = size(Z, 2);
    
    unq_max_Z = zeros(2, 1);
    unq_min_Z = zeros(2, 1);
    
    for i=1:n_z
        unq_max_Z(i) = Z_distribution(num2str(Z(unq_I(i), :))) + slack_ratio / 2 * Z_distribution(num2str(Z(unq_I(i), :)));
        unq_min_Z(i) = Z_distribution(num2str(Z(unq_I(i), :))) - slack_ratio / 2 * Z_distribution(num2str(Z(unq_I(i), :)));
    end


    %% we start by creating the coefficients for each variable A_ij - there are n_x*n_z such as that
    coefficients = zeros(n_x, n_z);
    
    
    %% indeed the coefficients are indicated by the dot product between <U' * X, V' * Z> -- this sort
    %% of mimics the value of singular values
    for i=1:n_x
        for j=1:n_z
            coefficients(i,j) = dot(V' * Z(unq_I(j), :)', U' * X(i, :)');
        end
    end
    
    %% reshape the coefficients into a vector -- because we assume a vector of n_x*n_z coefficients
    % %% for each A_ij - the variables are flattened
    f = reshape(coefficients', [1, n_x*n_z]);
    
    %% Now we are starting to set the constraints that enforce that each
    %% X has exactly one Z
    Aeq_constrain_each_x_to_1_z = zeros(n_x, n_x * n_z);
    
    for i=[1:n_x]
        %% take the relevant block over A_i[1,2,3,4,...n_z] and set it all to one, so we get sum A_ij for each i
        cols = [((i-1)*n_z+1):((i-1)*n_z + n_z)];
        Aeq_constrain_each_x_to_1_z(i, cols) = 1;
    end
    
    %% Now the beq for each Aeq_constrain_each_x_to_1_z needs to be one, so that sum A_ij = 1 for each i
    beq = ones(n_x, 1);
    Aeq = Aeq_constrain_each_x_to_1_z;
    
    %% now we set the "slack" for z, meaning, the upper bound and lower bound of how many edges can go into each z.
    %% we want two set of inequalities: for each i, sum_j A_ij <= upper_bound and for each i, sum_j A_ij >= lower_bound
    %% because we can only set <= inequalities, the lower_bound will be set by using sum_j -A_ij <= -lower_bound
    %% this means we have a total of 2*n_z inequalities
    slackA = zeros(n_z * 2, n_x * n_z);
    
    for i=1:n_x
        for j=1:n_z
            slackA(j, ((i-1)*n_z+j)) = 1;
            slackA(j + n_z, ((i-1)*n_z+j)) = -1;
        end
    end
    
    try
        %% now we set the actual slack bounds
        b = reshape([unq_max_Z; -unq_min_Z]', [n_z*2, 1]);
    catch
        disp("MAX")
        disp(unq_max_Z)
        disp("then min")
        disp(unq_min_Z)
        disp("n_z")
        disp(n_z)
    end
    
    % b = reshape([unq_max_Z; -unq_min_Z]', [n_z*2, 1]);
    
    % fprintf('n_x is %d, n_z is %d, unq_min_Z is %.2f, unq_max_Z is %.2f\nb is %.2f\n', n_x, n_z, unq_min_Z, unq_max_Z, b);
    
    % disp(['n_x is ', num2str(n_x), ', n_z is ', num2str(n_z),...
    %     ', unq_min_Z is ', num2str(unq_min_Z),...
    %     ', unq_max_Z is ', num2str(unq_max_Z)])
    % disp('b is ')
    % disp(num2str(b))
    
    %% this means that each A_ij should be between 0 and 1 (or actually, either 0 or 1)
    lb = zeros(n_x * n_z, 1);
    ub = ones(n_x * n_z, 1);
    
    %% this just indicates that we want *all* varibles to be 0/1
    intcon = (1:(n_x * n_z))';

    options = optimoptions('intlinprog','Display','off');
    E = intlinprog(-f,intcon,slackA,b,Aeq,beq,lb,ub, options);
    T = E;
    E = reshape(E, [n_z, n_x])';
    
    new_Z = zeros(n_x, n_d);
    for i=1:n_x
        I = find(E(i, :));
        new_Z(i, :) = Z(unq_I(I), :);
    end

end

%% Now G contains an n_x*n_z matrix that consists of 0/1 for whether there is
%% a connection between ith x and jth z or not.