import torch as th
from s3dg import S3D

# Instantiate the model
net = S3D('/home/madhekar/work/zsource/models/video_text/s3d_dict.npy', 512)

# Load the model weights
net.load_state_dict(th.load('/home/madhekar/work/zsource/models/video_text/s3d_howto100m.pth'))

# Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1] 
video = th.rand(2, 3, 32, 224, 224)

# Evaluation mode
net = net.eval()
 
# Video inference
video_output = net(video)
print('video output: ',video_output)
# Text inference
text_output = net.text_module(['open door', 'cut tomato'])
print('text output: ',text_output)


video_embedding = video_output['video_embedding']
print('video embedding: ',video_embedding)

text_embedding = text_output['text_embedding']
print('text embedding: ', text_embedding)
# We compute all the pairwise similarity scores between video and text.
similarity_matrix = th.matmul(text_embedding, video_embedding.t())

print('similarity matrix:', similarity_matrix)

