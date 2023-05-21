clc;
clear all;
close all;

%% Step 1: Read Image

[f,p] = uigetfile('*.jpg;*.bmp;*.tif');
[In,map] = imread([p f]);
he=imresize(In,[256 256]);
imshow(he), title('Input image');
Img_original=rgb2gray(he);
figure;
imshow(Img_original), title('gray image');

%% K-mean
num_cluster=3;
nrows = size(Img_original,1);
ncols = size(Img_original,2);
I_1D = reshape(Img_original,nrows*ncols,1);
[cluster_idx    mu]=kmeans(double(I_1D),num_cluster,'distance','sqEuclidean','Replicates',3);
[mu_sort id_sort]=sortrows(mu);

lookup = containers.Map(id_sort, 1:size(mu_sort,1));

cluster_idx_sort = lookup.values(num2cell(cluster_idx));
cluster_idx_sort = [cluster_idx_sort{:}];

pixel_labels = reshape(cluster_idx_sort,nrows,ncols);

segmented_images = cell(1,4);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:num_cluster
    color = he;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

figure,subplot(231);
imshow(pixel_labels,[]), title('image labeled by cluster index');

%% Plotting images
subplot(222);
imshow(segmented_images{1},[]), title('objects in cluster 1');

%%
subplot(223);
imshow(segmented_images{2},[]), title('objects in cluster 2');

%%
subplot(224);
imshow(segmented_images{3},[]), title('objects in cluster 3');

figure;
imshow(segmented_images{1});

I=rgb2gray(segmented_images{1});
figure;
imshow(I);
bw=im2bw(I,0.2);
figure;
imshow(bw);
bw1=bwareaopen(bw,500);
figure;
imshow(bw1);

offsets = [0 1; -1 1; -1 0; -1 -1];
GLCM2 = graycomatrix(I,'NumLevels',8,'Offset',offsets);
stats = GLCM_Features(GLCM2);

% Contrast

c = stats.contr;
Contrast = c(1)

%Correlation

cr = stats.corrm;
CoR = cr(1)

%Energy

E = stats.energ;
Energy = E(1)

%Discm

ds = stats.dissi;
Discm = ds(1)

% Homogeneity

H = stats.homom;
Homogeneity = H(1)

% Entropy

en = stats.entro(1)

% standard deviation
sd = std(std(double(I)))

% mean
mean1 = mean(double(I(:)))

data= [Contrast; CoR; Energy ;Discm;Homogeneity ;en; sd; mean1];


%% ANN Classification
load net1
y = round(sim(net1,data));

if y == 1
    msgbox('STARTING STAGE','Result');
elseif y == 2
    msgbox('MIDDLE STAGE','Result');
elseif y == 3
    msgbox('SEVERE STAGE','Result');
end



