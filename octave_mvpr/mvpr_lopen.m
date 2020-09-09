%MVPR_LOPEN - open list file for reading or writing
%
% fh = evex_lopen(fname, mode) open file fname for reading line by
% line. If the operation fails, error is thrown.
%
% Inputs:
%  'fname' - the file name
%  'mode'  - either 'read' or 'write' or 'append'
%            Write mode will truncate already existing file.
%            Mode can not be changed while the file is open.
% <Optional>
%  'comment'  - the comment character(s), default '%'
%  'delim'    - field delimiter character(s), default ' ' (space)
%
% Outputs:
%  'fh'    - evex_l file object is returned
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
% See also MVPR_LCLOSE.M, MVPR_LREAD.M, MVPR_LWRITE.M and
% MVPR_LWRITECOMMENT.M .
%
function fh = mvpr_lopen(fname, mode, varargin);

conf = struct( 'comment', '%',...
	'delim', ' ');
conf = mvpr_getargs(conf, varargin);

if ~ischar(conf.comment)
	error('invalid ''comment'' parameter');
end
if ~ischar(conf.delim)
	error('invalid ''delim'' parameter');
end

switch mode
	case 'read'
		fmode = 'rt';
	case 'write'
		fmode = 'wt';
	case 'append'
		fmode = 'a';
	otherwise
		error('unknown mode');
end

fid = fopen(fname, fmode);

if fid == -1
	error(['could not open file ' fname]);
end

fh = struct(...
	'fid', fid,...
	'fmode', fmode,...
	'fname', fname,...
	'comment', conf.comment,...
	'delim', conf.delim);
