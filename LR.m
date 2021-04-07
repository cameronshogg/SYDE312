clc
clear

%Load Sample Dataset
data = readmatrix('clean_data');

%Set Lambda for Ridge Regression Regularization 
ridge = 15;

%Load training dataset
X = readmatrix('X_train');
%Add intercept
X = [ones(length(X),1) X];
y = readmatrix('y_train');

%Load testing dataset
X_test = readmatrix('X_test');
%Add intercept
X_test = [ones(length(X_test),1) X_test];
y_test = readmatrix('y_test');

% %OLS: Solving the Normal Equations Directly
% L = inv(X'*X);
% beta_OLS = L*X'*y;

%OLS:Solving with QR Decomposition 
[Q,R] = qr(X,0);
beta_OLS = R\(Q'*y);

% %OLS:Solving with SVD
% [U,V,D] = svd(X,0);
% Dinv = diag(1./(diag(D)));
% beta_OLS = V*Dinv*U'*y;

%OLS: SSres on Training Set

y_pred = X*beta_OLS;
error = y - y_pred;
squared_error = error.^2;
sum(squared_error)

%OLS: RMSE
y_pred = X_test*beta_OLS;
error = y_test - y_pred;
squared_error = error.^2;
sqrt(mean(squared_error))

%Ridge Regression: Solving the Normal Equations Directly
L = inv(X'*X+ridge*eye(length(X'*X)));
beta_RR = L*X'*y;

%Ridge Regression: RMSE
y_pred = X_test*beta_RR;
error = y_test - y_pred;
squared_error = error.^2;
sqrt(mean(squared_error))

%Ridge Regression: SSres on Training Set
y_pred = X*beta_RR;
error = y - y_pred;
squared_error = error.^2;
sum(squared_error)

mean(error)



