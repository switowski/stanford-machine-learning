function [J, grad] = costFunctionReg(theta, X, y, lambda)
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters.

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly
    J = 0;
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    function h = sig (_theta, _x)
        % Helper function for vectorized sigmoid
        % We need to transpose x here to make it a vector, so the next line of
        % code has 2 transposing operations, not one, like in the videos/slides/
        h = 1/(1 + e^-(_theta' * _x'));
    end

    % Let's use the loop this time
    % COST FUNCTION
    for i=1:m
        J += y(i)     * log(    sig(theta, X(i,:))) + ...
             (1-y(i)) * log(1 - sig(theta, X(i,:)));
    end
    J = (-1/m) * J;
    r = 0;
    % Add the regularization parameter
    for j=2:size(X,2)
        r += theta(j)^2;
    end
    r = lambda / (2*m) * r;

    % Final, regularized version of cost function
    J += r;

    % GRADIENT
    for j=1:size(theta)
        g = 0;
        for i=1:m
            g += (sig(theta, X(i,:)) - y(i)) * X(i,j);
        end
        grad(j) = 1/m * g;
        if j > 1
            grad(j) += (lambda / m) * theta(j);
        end
    end

    % =============================================================

end
