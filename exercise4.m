function exercise4(varargin)
% EXERCISE4   Part 4 of the VGG CNN practical

setup('useGpu', true) ;
mkdir(['/tmp/' getenv('USER')]) ;

% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Create checkpoint directories
mkdir(['/tmp/' getenv('USER') '/charcnn']) ;
mkdir(['/tmp/' getenv('USER') '/charscnn-jit']) ;

% Load character dataset
imdb = load('data/charsdb.mat') ;

% Visualize some of the data
figure(1) ; clf ; colormap gray ;
subplot(1,2,1) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==1)) ;
axis image off ;
title('training chars for ''a''') ;

subplot(1,2,2) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==2)) ;
axis image off ;
title('validation chars for ''a''') ;

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = initializeCharacterCNN() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 15 ;
trainOpts.continue = true ;
trainOpts.gpus = [1] ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = ['/tmp/' getenv('USER') '/charscnn'] ;
%trainOpts = vl_argparse(trainOpts, varargin);

% Take the average image out
imdb = load('data/charsdb.mat') ;
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;

% Convert to a GPU array if needed
if numel(trainOpts.gpus) > 0
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to the CPU if it was trained on the GPU
if numel(trainOpts.gpus) > 0
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save(['/tmp/' getenv('USER') '/charscnn.mat'], '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.weights{1}),'spacing',2)
axis equal ; title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Part 4.5: apply the model
% -------------------------------------------------------------------------

% Load the CNN learned before
net = load(['/tmp/' getenv('USER') '/charscnn.mat']) ;
%net = load(['tmp/' getenv('USER') '/charscnn-jit.mat']) ;

% Load the sentence
[im,cmap] = imread('data/sentence-lato.png') ;
if isempty(cmap)
  im = im2single(im) ;
else
  im = im2single(ind2gray(p,cmap)) ;
end
im = 256 * (im - net.imageMean) ;

% Apply the CNN to the larger image
res = vl_simplenn(net, im) ;

% Visualize the results
figure(3) ; clf ;
decodeCharacters(net, imdb, im, res) ;

% -------------------------------------------------------------------------
% Part 4.6: train with jitter
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 15 ;
trainOpts.continue = true ;
trainOpts.gpus = [1] ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = ['/tmp/' getenv('USER') '/charscnn-jit'] ;

% Initlialize a new network
net = initializeCharacterCNN() ;

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatchWithJitter, trainOpts) ;

% Move the CNN back to CPU if it was trained on GPU
if numel(trainOpts.gpus) > 0
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save(['/tmp/' getenv('USER') '/charscnn-jit.mat'], '-struct', 'net') ;

% Visualize the results on the sentence
figure(4) ; clf ;
decodeCharacters(net, imdb, im, vl_simplenn(net, im)) ;
