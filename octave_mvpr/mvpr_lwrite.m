%MVPR_LWRITE - write a line of data into a list file
%
% [] = mvpr_lwrite(fh, data) will write the contents of 'data' into
% opened list file, separating different cells with the first
% delimiter character defined in mvpr_lopen. 
%
% Inputs:
%  'fh'   - evex_l file object created by evex_lopen
%  'data' - string or cell array of strings
%
% Outputs:
%  -
%
% Examples:
%  -
%
% Authors:
%  Pekka Paalanen, MVPR in 2009
%
% Project:
%  -
%
% References:
%
% See also MVPR_LCLOSE.M, MVPR_LOPEN.M, MVPR_LREAD.M, and
% MVPR_LWRITECOMMENT.M .
%
function mvpr_lwrite(fh, data);

if fh.fmode(1) ~= 'w' && fh.fmode(1) ~= 'a'
	error('trying to write a file opened for reading');
end

if ~iscell(data)
	data = {data};
end

len = length(data);
for i = 1:len
	if i==len
		fprintf(fh.fid, '%s\n', data{i});
	else
		fprintf(fh.fid, '%s%c', data{i}, fh.delim(1));
	end
end
