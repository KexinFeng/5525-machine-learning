function tf = allclose(a, b, err)
    tf = all(abs(a(:) - b(:)) < err);
end
