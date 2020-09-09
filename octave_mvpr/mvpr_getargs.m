%MVPR_GETARGS  parse variable argument list into a struct
%
% S = MVPR_GETARGS(defaultS, varglist) can be conventiently used
% for parsing variable list of arguments with default values for
% any function (see MVRP_LOPEN.M for example).
%
% Inputs:
%  varglist - a cell array of name, value pairs, or
%             a struct containing some updated values
%  defaultS - struct containing the default values
%
% Outputs:
%  S - Updated structure containing the variables.
%
% Examples:
%   function foo(par1, varargin);
%    args = struct( 'param1', 0, 'param2', eye(3) );
%    args = getargs( args, varargin );
%    disp(args.param1);
%
%   foo(2, 'param1', 14) will print 14
%
% Authors:
%   Pekka Paalanen <paalanen@lut.fi>
%
% Project:
%  - 
%
% References:
%
% -
%
% See also .
%
function S = mvpr_getargs(defaultS, varglist);

if length(varglist) == 1 && isa(varglist{1}, 'struct')
	% We got an argument structure instead of varargin
	S = getargs_struct(defaultS, varglist{1});
elseif mod(length(varglist), 2) == 0
	S = getargs_cell(defaultS, varglist);
else
	error('Parent function called with invalid number of arguments.');
end


% ----------------------------------------------------------
function S = getargs_struct(defaultS, vargs);

S = defaultS;

fields = fieldnames(vargs);
for i = 1:length(fields);
	if isfield(S, fields{i})
		% for Matlab R12
		%S = setfield(S, fields{i}, getfield(vargs, fields{i}));
		
		% for Matlab R13 and above
		S.(fields{i}) = vargs.(fields{i});
	else
		mvpr_warning_wrap('getargs:unknown_param', ...
                                  ['Unknown parameter "' fields{i} '"']);
	end
end

% ----------------------------------------------------------
function S = getargs_cell(defaultS, varglist);

S = defaultS;
i=1;
while i <= length(varglist)
	if isfield(S, varglist{i})
		% for Matlab R12
		%S = setfield(S, varglist{i}, varglist{i+1});
		
		% for Matlab R13 and above
		S.(varglist{i}) = varglist{i+1};
	else
		mvpr_warning_wrap('getargs:unknown_param', ...
                                  ['Unknown parameter "' varglist{i} '"']);
	end
	i = i+2;
end
