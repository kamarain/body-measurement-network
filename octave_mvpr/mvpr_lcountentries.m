%MVPR_LCOUNTENTRIES - Counts how many valid lines in the given file
%
% c = mvpr_lcountentries(fname_) opens the file fname and counts
% how many mvpr_lread:able lines it contains (usefull for progress
% printing for-loops, for example).
%
% Inputs:
%  fname - File name.
% 
%
% Outputs:
%  c     - Total count.
%
% Examples:
%  -
%
% Authors:
%  Joni Kamarainen, MVPR in 2009
%
% Project:
%  -
%
% References:
%
% See also MVPR_LOPEN.M, MVPR_LCLOSE.M, MVPR_LREAD.M, MVPR_LWRITE.M and
% MVPR_LWRITECOMMENT.M .
%
function [c] = mvpr_lcountentries(fname_,varargin)

conf = struct( 'comment', '%');
conf = mvpr_getargs(conf, varargin);

fh = mvpr_lopen(fname_, 'read','comment',conf.comment);
filepair = mvpr_lread(fh);
c = 0;
while ~isempty(filepair)
    c = c+1;
    filepair = mvpr_lread(fh);
end;
mvpr_lclose(fh);

