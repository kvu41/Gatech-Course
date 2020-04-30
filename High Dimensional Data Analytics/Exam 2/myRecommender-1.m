function [ U, V ] = myRecommender( rateMatrix, lowRank )

    % Parameters
    maxIter = 0; % Choose your own.
    learningRate = 0; % Choose your own.
    regularizer = 0; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    % Gradient Descent:
    
    % IMPLEMENT YOUR CODE HERE.
end