%MVPR_LREAD - read a line from list file
%
% [data] = mvpr_lread(fh) will read the next line from an opened
% list file, skipping comment and empty lines. Comments begin at
% the specified comment character and continue to the end of the
% line. Empty lines consist of only white space characters (after
% comment removal).
%
% On end of file evex_lread will return empty.
%
% Inputs:
%  'fh'   - mvpr_l file object created by mvpr_lopen
%
% Outputs:
%  'data' - cell array of strings, items on a line
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
% See also MVPR_LOPEN.M, MVPR_LCLOSE.M, MVPR_LWRITE.M and
% MVPR_LWRITECOMMENT.M .
%
function data = mvpr_lread(fh);

if fh.fmode(1) ~= 'r'
	error('trying to read a file opened for writing');
end

data = {};

while 1
	% read a line
	line = fgetl(fh.fid);
	
	if isempty(line)
		continue;
	end
	
	% EOF?
	len = length(line);
	if (len == 1) && (line == -1)
		return;
	end
	
	% purge comments
	if ~isempty( strfind(fh.comment, line(1)) )
		continue;
	end
	line = strtok(line, fh.comment);
	
	% skip empty lines
	if ~all(isspace(line))
		break;
	end
end

% we have a line, chop it
i=1;
while ~isempty(line)
	[item, line] = strtok(line, fh.delim);
	data{i} = item;
	i=i+1;
end
