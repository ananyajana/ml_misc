from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
from sklearn.datasets import load_sample_image
from scipy.misc import imsave
from skimage import io
#image = load_sample_image('flower.jpg')
#image = Image.open('flower.jpg')
image = Image.open('cat.jpg').convert('RGB')
imshow(image)

#img = io.imread('12.png')
#image = Image.fromarray(img.astype('uint8'), 'L').convert('RGB')
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    #transforms.Resize((224, 224)),
    #transforms.Resize((512, 512)),
    transforms.ToTensor(),
    normalize
    ])

display_transform = transforms.Compose([
    transforms.Resize((224, 224))])

tensor = preprocess(image)
prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad = True)

model = models.resnet18(pretrained=True)
print(model)
model.cuda()
model.eval()

class save_features():
    features = None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

final_layer = model._modules.get('layer4')
activated_features = save_features(final_layer)

prediction = model(prediction_var)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()
#print('activated_features.features :', activated_features.features)
#print('activated_features.features shape :', activated_features.features.shape)
#print('activated_features.features type :', type(activated_features.features))
# checking how much confident our model is that this picture is class 283
#print('pred probabilities are :', pred_probabilities)

print(topk(pred_probabilities, 1))

def get_CAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    print('feature_conv.shape', feature_conv.shape)
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    # the following two lines are for normalization
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]
    return [cam]

weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
print('weight_softmax_params len :', len(weight_softmax_params))
print('weight_softmax_params[0] shape :', (weight_softmax_params[0].shape))
#print('the named_parameters in the model')
#for name, param in model._modules.get('fc').parameters():
#print('count of named parameters :', len(list(model.named_parameters())))
#for name, param in model.named_parameters():
#    print(name)
'''
print('weights softmax ', weight_softmax_params[0])
print('weights softmax ', (weight_softmax_params[0]).size())

print('weights softmax np squeeze', np.squeeze(weight_softmax_params[0]))
print('weights softmax np squeeze', np.squeeze((weight_softmax_params[0])).size())

print('the named_parameters in fc layer')
print('count of named parameters :', len(list(model._modules.get('fc').named_parameters())))
for name, param in model._modules.get('fc').named_parameters():
    print(name)
'''
    
#print('the parameters in the model')
#for name, param in model._modules.get('fc').parameters():
#print('count of parameters :', len(list(model.parameters())))
#for param in model.parameters():
#    print(param)
'''
print('the parameters in fc layer')
print('count of named parameters :', len(list(model._modules.get('fc').named_parameters())))
for param in model._modules.get('fc').parameters():
    print(param)
''' 
#class_idx = topk(pred_probabilities, 1)[1].int()
class_idx = topk(pred_probabilities, 10)[1][2].int()
print('1. class idx is :', class_idx)
print('topk of pred probabilities are :', topk(pred_probabilities, 10))
#print('topk of pred probabilities are :', topk(pred_probabilities, 2)[1][0].int())
'''
from torch import ones, int32
class_idx = ones([1], device='cuda:0', dtype=int32)
print('1. class idx is :', class_idx)
class_idx[0] = 5
print('1. class idx is :', class_idx)
'''
'''
print('weight_softmax[idx] shape :', (weight_softmax[class_idx]).shape)
print('weight_softmax[class_idx] :', weight_softmax[class_idx])
print('weight_softmax[class_idx] type:', type(weight_softmax[class_idx]))
print('weight_softmax[class_idx][0] :', weight_softmax[class_idx][0])
_, nc, h, w = activated_features.features.shape
print('activated_features.features reshaped shape :', activated_features.features.reshape((nc, h*w)).shape)
'''
overlay = get_CAM(activated_features.features, weight_softmax, class_idx)
'''
print('overlay :', overlay[0])
print('overlay.shape :', overlay[0].shape)
print('overlay type :', type(overlay[0]))
'''
#imsave('cam_view.png', overlay)
#imshow(overlay[0], alpha = 0.5, cmap = 'jet')
#imshow(overlay[0], alpha = 0.5, cmap = 'viridis')
#display_transform(image)
#imshow(image)

imshow(display_transform(image))
#imshow(overlay[0], alpha = 0.5, cmap = 'jet')
imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha = 0.5, cmap = 'jet')
#imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha = 0.5)
#import matplotlib
#imsave('new.png', skimage.transform.resize(overlay[0], tensor.shape[1:3]), cmap ='jet')
#print('pred probabilities are :', pred_probabilities)
#print('pred probabilities size :', pred_probabilities.size())
#print('topk of pred probabilities are :', topk(pred_probabilities, 1)[1].int())
#print('topk of pred probabilities are :', topk(pred_probabilities, 2)[1][1].int())

'''
class_idx = topk(pred_probabilities, 2)[1][1].int()
print('2. class idx is :', class_idx)

overlay = get_CAM(activated_features.features, weight_softmax, class_idx)
imshow(display_transform(image))
imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha = 0.5, cmap = 'jet')
'''