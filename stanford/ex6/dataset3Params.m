function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_ = [.01 .03 .1 .3 1 3 10 30];
sigma_ = [.01 .03 .1 .3 1 3 10 30];

min_error = 1e6;
Cmin = C_(1);
sigmamin = sigma_(1);
iter = 0;

for i = 1:length(C_)
  for j = 1:length(sigma_)

    model = svmTrain(X, y, C_(i), @(x1, x2) gaussianKernel(x1, x2, sigma_(j)));
    
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));

    if error < min_error
      min_error = error;
      Cmin = C_(i);
      sigmamin = sigma_(j);
    endif
  
    iter ++;
    fprintf(["==== %f %f %f %f"], iter, i, j, error);
  
  endfor
endfor

C = Cmin;
sigma = sigmamin;

fprintf(["Min: %f %f %f"], C, sigma, min_error);

% =========================================================================

end
