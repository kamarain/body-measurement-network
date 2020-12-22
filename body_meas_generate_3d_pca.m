% Configuration for data generation
temp_work='./TEMPWORK'
num_of_pca = 50
generate_num = 100
verbose = 0
data_dir = 'caesar-fitted-meshes'
data_file = 'TEMPWORK/mpii_caesar_male_MAT_TRAIN.txt'

% MVPR helper functionality
addpath('octave_mvpr/');

% Read data
num_of_samples = mvpr_lcountentries(data_file)

fid = mvpr_lopen(data_file,'read');

P = [];
for i = 1:num_of_samples
  fprintf('\r  Read data %4d/%4d',i,num_of_samples)
  if exist('OCTAVE_VERSION', 'builtin')
    fflush(stdout);
  end;
  sample_file = mvpr_lread(fid);
  sample = load(fullfile(data_dir,sample_file{1}));
  sample = sample.points;
  sample_row = reshape(sample',[1 size(sample,1)*3]);
  P = [P; sample_row];
end
fprintf(' Done!\n')

fprintf('Computing PCA singular values and vectors...')
if exist('OCTAVE_VERSION', 'builtin')
  fflush(stdout);
end;
% Center data
[m,n] = size(P);
i     = ones(m,1);

P_mean = mean(P);
cP     = P - P_mean(i,:);

% Compute PCA for centered data
[U S V] = svd(cP,0);
pcaProjection = U*S;
pcaP = pcaProjection(:,1:num_of_pca);
pcaV = V(:,1:num_of_pca);
e = diag(S).^2/size(S,1);
fprintf(' Done!\n')
if exist('OCTAVE_VERSION', 'builtin')
  fflush(stdout);
end;


% Create dir where to store generated samples
[foo,fname,foo] = fileparts(data_file);
save_dir = fullfile(temp_work,[fname '_GENERATED']);
if ~exist(save_dir,'dir')
  mkdir(save_dir)
end;

% File that lists generated samples
[foo,fname,foo] = fileparts(data_file);
save_list_file = fullfile(temp_work,[fname '_GENERATED.txt']);
save_fid = mvpr_lopen(save_list_file,'write');

for i = 1:generate_num
  fprintf('\r  Generating samples %4d/%4d',i,generate_num)
  if exist('OCTAVE_VERSION', 'builtin')
    fflush(stdout);
  end;
  % Create 1D Gaussian random numbers according to the data
  e_r = normrnd(zeros(num_of_pca,1),sqrt(e(1:num_of_pca)));
  new_sample = V(:,1:num_of_pca)*e_r+P_mean';
  new_sample_3d = reshape(new_sample, [3 size(sample,1)])';
  if verbose
    clf
    plot3(new_sample_3d(1:10:end,1),new_sample_3d(1:10:end,2), ...
          new_sample_3d(1:10:end,3),'ko');
    fprintf('Generated person height = %3.2f cm', (max(new_sample_3d(:,3))-min(new_sample_3d(:,3)))/10);
    if exist('OCTAVE_VERSION', 'builtin')
      fflush(stdout);
    end;
    input('<RETURN>')
  end;
  save_file = fullfile(save_dir, sprintf('%07d.mat',i));
  synth_sample.points = new_sample_3d;
  if exist('OCTAVE_VERSION', 'builtin')
    save('-mat-binary',save_file,'synth_sample')
  else
    save(save_file,'synth_sample')
  end;
  mvpr_lwrite(save_fid,save_file);
end;
mvpr_lclose(save_fid);
fprintf(' Done!\n')
if exist('OCTAVE_VERSION', 'builtin')
  fflush(stdout);
end;

fprintf('All generated samples stored to %s\n (see also %s)\n', ...
        save_dir,save_list_file);
if exist('OCTAVE_VERSION', 'builtin')
  fflush(stdout);
end;
