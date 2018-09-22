function [J, grad] = costFunction(theta, X, y)
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.

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
    %
    % Note: grad should have the same dimensions as theta
    %

    function h = sig (_theta, _x)
        % Helper function for vectorized sigmoid
        % We need to transpose x here to make it a vector, so the next line of
        % code has 2 transposing operations, not one, like in the videos/slides/
        h = 1/(1 + e^-(_theta' * _x'));
    end
    % Let's use the loop this time
    for i=1:m
        J += y(i)     * log(    sig(theta, X(i,:))) + ...
             (1-y(i)) * log(1 - sig(theta, X(i,:)));
    end
    J = (-1/m) * J;

    for j=1:size(theta)
        g = 0;
        for i=1:m
            g += (sig(theta, X(i,:)) - y(i)) * X(i,j);
        end
        grad(j) = 1/m * g;
    end


% =============================================================

end
