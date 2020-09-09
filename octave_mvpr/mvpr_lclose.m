%MVPR_LCLOSE - close list file
%
% [] = mvpr_lclose(fh) closes the file. If 'fh' is valid, this
% function cannot fail.
%
% Inputs:
%  'fh' - the mvpr_l file object created with mvpr_lopen.
%
% Outputs:
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
% See also MVPR_LOPEN.M, MVPR_LREAD.M, MVPR_LWRITE.M and
% MVPR_LWRITECOMMENT.M .
%
function mvpr_lclose(fh);

if( fclose(fh.fid) == -1 )
	warning(['closing file ' fh.fname ' failed.']);
end
