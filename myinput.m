function [value] = myinput(promt, default)
%MYINPUT Does the same as MATLAB.Input, but returns a default value.
value = input(promt);
if exist('default', 'var') && isempty(value) 
    value = default;
end
end

