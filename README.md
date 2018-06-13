# Neural-Image-Captioning-using-transfer-learning
Image captioning uses techniques from computer vision and natural language processing to predict the content of the images. To implement this, we train the model integrating both the subfields using various datasets. Datasets usually contains a plethora of images, which is both computationally expensive and time consuming to train. We propose a scheme to employ transfer learning in pre-trained models  to reduce the complexity of the training phase in new datasets. This report also explains about how we used transfer learning to a model which performs a different task, to make it perform our task. Experiments on  various datasets have been conducted and results have been tabulated. We analyze the accuracy of the model both qualitatively and quantitatively.

Pre-Trained Models used
1. Pre-trained Flickr8k Model: One of the pre-trained
model chosen by us was pre-trained Flickr8k model.
This model is being trained upon Flickr8k dataset,
which consists of 8000 images obtained from the
Flickr website. In this dataset, it includes five descriptions
per each image. This pre-trained model was
chosen as reference for the function of image captioning
task on new datasets. This was achieved through
transfer learning. After applying transfer learning, we
tested upon Flickr30k and Pascal dataset and obtained
good results, which is shown in the results section below.

2. Pre-trained VQG Model: VQG: Visual Question Generator.
As there was no pre-trained model readily available
for VQG, we trained the model from scratch to
generate questions as the output upon image as input. This pre-trained model is trained on VQG MS COCO
5000 dataset, which consists of 5000 images of MS
COCO, which were also annotated by the CQA dataset
and by VQA. This dataset also consists of 5 descriptions
per image.
Using this model was one of the challenging task of
our project where the model that is pre-trained to perform
another task (question generation), is modified to
perform our task, image caption generation, by applying
the process of transfer learning to the model.

Pre-processing of data:
Photo data: The pre-processing of images involved resizing them to
a size of 224 x 224 as per VGG model requirements. The
images were also zero-centered and normalized for better
image representations. We used a pre-trained VGG model
to interpret the content of the photos. We pre-computed
the photo features using the pre-trained model and saved
them to a file for later use as running each photo through the
network every time we want to test a new language model
configuration will increasing our training time and consume
more memory. The last layer of the model is removed as the
model is used to predict a classification for a photo. Since
we’re not interested in classification, we stop at the previous
fully connected layer and extract the image features which
are a 1-dimensional 4,096 element vector.

Text data: Flickr8k dataset contains a text file which contains multiple
descriptions for each photograph. Each photo has a
unique identifier and the descriptions are already tokenized.
Each photo identifier maps to a list of one or more textual
descriptions. Ideally, we want a vocabulary that is both expressive
and as small as possible. We also reduced the size
of the vocabulary of words by converting all words to lowercase,
removing all punctuation and removing all words that
are 1 character long. This helped us to get a vocabulary that
is both expressive and small which also helped us to train
faster

Pre-trained VQG model: Since VQG pre-trained model
was designed to generate questions, we had to do certain
pre-processing in order to achieve our task of captioning.
The pre-processing of questions involved removing all stop
words and punctuation marks in the questions. Further, the
questions were post-padded to get a consistent input for the
model and delimiters were added to beginning and end of
each sentence.

Model:
Our model has 3 main components which are as follows:
1. Photo Feature Extractor: We have pre-processed the
photos with the VGG model (removed the output layer
as explained before) and we used the extracted features
(4096 element vector) predicted by this model as input.
This is a 16-layer VGG model pre-trained on the ImageNet
dataset.
2. Sequence Processor: First, we have a word embedding
layer for handling the text input, followed by a Long
Short-Term Memory (LSTM) recurrent neural network
layer.
3. Decoder: The output of the feature extractor and sequence
processor gives us a fixed-length vector. This
is then fed to a Dense 256 neuron layer and then to a final
output Dense layer that makes a softmax prediction
over the entire output vocabulary for the next word in
the sequence.
In this section, we describe our model that was used
for generating questions from images. We made use
of the VGG-Net(Simonyan Zisserman, 2014) architecture
to extract representations from images. Representations
from the pre-final fully connected layer, fc7,
were used as one of the two inputs for generating questions.
The representations extracted by the fc7 layer
are 4096-dimensional. These representations are then
passed through a fully connected layer with 256 neurons.
The input questions, which is the second input,
are passed through an embedding layer of 256 dimensions.
The word embeddings are then passed into a
LSTM with 256 cells. The two outputs, from LSTM
and fully connected layer with fc7 inputs, are combined
and passed through a dense layers with 256 neurons.
This layer is connected to the output layer that
has a size which is equivalent to the vocabulary size.
Our model is trained using an ADAM optimizer on a
categorical cross entropy loss. The maximum length
of the generated question was chosen to be 34 based
on dataset statistics. The dropout value for the LSTM
layers was set to 0.5 while the dropout for the dense
layer with fc7 inputs was set to 0.5. The initial learning
rate of the model was set to 0.001 with a weight
decay of 0.5. The model was trained for 20 epochs.

