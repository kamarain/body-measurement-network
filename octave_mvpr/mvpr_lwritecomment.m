%MVPR_LWRITECOMMENT - write a comment line into a list file
%
% mvpr_lwritecomment(fh, comment) will write the comment string
% into opened list file, prefixing it with the first comment
% character defined in mvpr_lopen. 
%
% Inputs:
%  'fh'      - evex_l file object created by evex_lopen
%  'comment' - string
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
% MVPR_LWRITE.M .
%
function mvpr_lwritecomment(fh, data);

if fh.fmode(1) ~= 'w'
	error('trying to write a file opened for reading');
end

fprintf(fh.fid, '%c%s\n', fh.comment(1), data);
