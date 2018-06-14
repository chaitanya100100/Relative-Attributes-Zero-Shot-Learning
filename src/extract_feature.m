%im_path = '../saved_data/real_images/Avinash_Sharma.jpg';
%out_path = '../saved_data/real_images/Avinash_Sharma.mat';

im_path = '../../relative_attributes/pubfig/images/AlexRodriguez_1.jpg';
out_path = '../saved_data/real_images/AlexRodriguez_1.mat';


img = imread(im_path);
gist_feat = extract_gist(img);
imglab = rgb2lab(img);
hist_feat = imhist(imglab, 30) / numel(imglab);

tot_feat = [gist_feat, hist_feat'];
save(out_path, 'tot_feat');
