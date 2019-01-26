function tf = allclose(a, b, err)
    a = double(a);
    b = double(b);
    tf = all(abs(a(:) - b(:)) < err);
end
