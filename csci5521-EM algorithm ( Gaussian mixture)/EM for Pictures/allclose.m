function tf = allclose(a, b, err)
if nargin < 3
    err = 1e-6;
end
    tf = all(abs(a(:) - b(:)) < err);
end
