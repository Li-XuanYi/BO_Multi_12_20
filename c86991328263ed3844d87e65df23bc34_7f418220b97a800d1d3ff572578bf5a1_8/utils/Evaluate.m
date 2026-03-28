function f = Evaluate(x)
    [m,d] = size(x);
    x_m = x(:,2:end);
    g = 1*(d - 2 + 1 + sum((x_m - 0.5).^2 - cos(2*pi*(x_m - 0.5)),2));
    f1 = (1 + g).*cos(pi*x(:,1)/2);
    f2 = (1 + g).*sin(pi*x(:,1)/2);
    f = [f1,f2];
end