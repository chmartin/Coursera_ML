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

% Selected values of C and sigma
test_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

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

best_train = 1;
best_val = 1;

for i = 1:length(test_vec) %loop over C
	my_C = test_vec(i);
	for j = 1:length(test_vec) %loop over sigma
		my_sigma = test_vec(j);
		
		%train with these parameters
		model= svmTrain(X, y, my_C, @(x1, x2) gaussianKernel(x1, x2, my_sigma));
		
		%Compute the training / Validation errors with this C and sigma
		predict_train = svmPredict(model,X);
		predict_val = svmPredict(model,Xval);
		error_train = mean(double(predict_train ~= y));
		error_val = mean(double(predict_val ~= yval));
		%error_val
		%Update C and sigma only if this performs better on the validaiton set
		if(error_val < best_val)
			best_val = error_val;
			%best_val
			C = my_C;
			sigma = my_sigma;
		endif
	end
end

%C
%sigma


% =========================================================================

end
