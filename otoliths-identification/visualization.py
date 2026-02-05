import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import os
import numpy as np
import argparse
import torch.nn as nn
import config as conf
from model.CNNs import FineTuneModel_Hierarchical, FineTuneModel_Hierarchical_Attention
import sys
import shutil
import glob
from util.useful_imports import copyfile


class FeatureExtractorAttention():
    """
    Feature extractor for Attention ResNet18
    Properly handles the complete forward pass through attention blocks
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        # Forward through initial layers
        x = self.model.conv1(x)
        if '0' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        x = self.model.bn1(x)
        if '1' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        x = self.model.relu(x)
        if '2' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        x = self.model.maxpool(x)
        if '3' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        # Forward through attention layers
        x = self.model.layer1(x)
        if '4' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        x = self.model.layer2(x)
        if '5' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        x = self.model.layer3(x)
        if '6' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        x = self.model.layer4(x)
        if '7' in self.target_layers:
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        # CRITICAL: Apply avgpool and flatten
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return outputs, x


class FeatureExtractor():
    """
    Standard feature extractor for non-attention models
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """
    Unified model output handler for both attention and standard models
    """
    def __init__(self, model, target_layers, is_attention=False):
        self.model = model
        self.is_attention = is_attention
        
        if is_attention:
            self.feature_extractor = FeatureExtractorAttention(
                self.model, target_layers)
        else:
            self.feature_extractor = FeatureExtractor(
                self.model.features, target_layers)
    
    def get_gradients(self):
        return self.feature_extractor.gradients
    
    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        
        # For non-attention models, flatten output
        if not self.is_attention:
            output = output.view(output.size(0), -1)
        
        # Apply dropout if present
        if hasattr(self.model, 'dropout') and self.model.dropout is not None:
            output = self.model.dropout(output)
        
        # Pass through classifier
        output = self.model.classifier(output)
        
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, filename):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(filename, np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda, is_attention=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.is_attention = is_attention
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names, is_attention)
    
    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        # Zero gradients
        self.model.zero_grad()
        
        one_hot.backward(retain_graph=True)
        
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) > 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(
            input.size()).type_as(input), input, positive_mask)
        ctx.save_for_backward(input, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = None
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(
            torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda, is_attention=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.is_attention = is_attention
        
        if self.cuda:
            self.model = model.cuda()
        
        # Replace ReLU with GuidedBackpropReLU
        if not is_attention:
            for idx, module in self.model.features._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    self.model.features._modules[idx] = GuidedBackpropReLU.apply
    
    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)
        
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        one_hot.backward(retain_graph=True)
        
        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        
        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                       help='Input image path')
    parser.add_argument('-g', type=str, default=1, help='Group')
    parser.add_argument('--use_attention', dest='use_attention', default=False,
                       action='store_true', help='use attention ResNet18')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    return args


class MyCNNs(nn.Module):
    """
    Wrapper model for Grad-CAM visualization
    Works with both standard and attention models
    """
    def __init__(self, group=1, use_attention=False):
        super(MyCNNs, self).__init__()
        model, is_attention = self.get_model(use_attention)
        self.is_attention = is_attention
        
        # Store model components
        if is_attention:
            # For attention model
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.avgpool = model.avgpool
        else:
            # For standard model
            self.features = model.features
        
        self.classifier = model.level_1_0 if group == 1 else model.level_1_1
        self.dropout = model.dropout if hasattr(model, 'dropout') else None
    
    def forward(self, x):
        if self.is_attention:
            # Manual forward for attention model
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            f = torch.flatten(x, 1)
            if self.dropout:
                f = self.dropout(f)
        else:
            f = self.features(x)
            f = f.view(f.size(0), -1)
        
        y = self.classifier(f)
        return y
    
    def get_model(self, use_attention=False):
        """Load model and auto-detect if attention"""
        base_model = models.resnet18(pretrained=True)
        
        # Load checkpoint
        checkpoint = torch.load(
            './output/best_resnet18.pth.tar',
            map_location=lambda storage, loc: storage,
            weights_only=False
        )
        
        # Detect model type from checkpoint
        state_dict_keys = checkpoint['state_dict'].keys()
        is_attention = any('cbam' in key or 'channel_attention' in key 
                          for key in state_dict_keys)
        
        # Override with command line if specified
        if use_attention or is_attention:
            print("Loading Attention ResNet18 model")
            model = FineTuneModel_Hierarchical_Attention(base_model, 'resnet18', None)
            is_attention = True
        else:
            print("Loading Standard Hierarchical model")
            model = FineTuneModel_Hierarchical(base_model, 'resnet18', None)
            is_attention = False
        
        model.load_state_dict(checkpoint['state_dict'])
        model.args = checkpoint['args']
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()
        model.eval()
        
        return model, is_attention


def make_sample(sample_dir):
    """Create sample directory with one image per species"""
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.mkdir(sample_dir)
    
    for label in os.listdir(conf.TRAIN_DIR):
        tar_dir = os.path.join(conf.TRAIN_DIR, label)
        img_list = glob.glob(tar_dir + '/*.jpg')
        if len(img_list) > 0:
            selected = np.random.randint(len(img_list))
            copyfile(img_list[selected], os.path.join(
                sample_dir, label + '.jpg'))
    
    print(f"Created {len(os.listdir(sample_dir))} sample images in {sample_dir}")


if __name__ == '__main__':
    """
    Generate Grad-CAM visualizations for all layers
    Compatible with both standard and attention ResNet18
    """
    sample_dir = './output/sample/'
    args = get_args()
    
    # Create sample images
    make_sample(sample_dir)
    
    # Labels of group 0 (Sandeels) and group 1 (Clupeids)
    gr_0_lab = ["Ammodytes tobianus", "Hyperoplus lanceolatus", "Ammodytes marinus"]
    gr_1_lab = ["Clupea harengus", "Sprattus sprattus", "Alosa fallax"]
    
    # Create output directories for each layer
    for layer in range(8):
        out_dir = os.path.join(sample_dir, str(layer + 1))
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    
    print("\nStarting Grad-CAM visualization...")
    
    # Process each species
    for label in gr_0_lab + gr_1_lab:
        args.g = int(label in gr_1_lab) + 1
        args.image_path = os.path.join(sample_dir, label + '.jpg')
        
        print(f"Processing {label}...")
        
        # Generate Grad-CAM for each layer
        for layer in range(8):
            out_dir = os.path.join(sample_dir, str(layer + 1))
            
            # Create model for this group with attention support
            model_cnn = MyCNNs(group=args.g, use_attention=args.use_attention)
            is_attention = model_cnn.is_attention
            
            grad_cam = GradCam(
                model=model_cnn,
                target_layer_names=[str(layer)],
                use_cuda=args.use_cuda,
                is_attention=is_attention
            )
            
            img = cv2.imread(args.image_path, 1)
            if img is None:
                print(f"Warning: Could not read {args.image_path}")
                continue
            
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(os.path.join(out_dir, label + '.jpg'), img)
            
            img = np.float32(img) / 255
            input = preprocess_image(img)
            
            # Generate CAM
            target_index = None
            try:
                mask = grad_cam(input, target_index)
                show_cam_on_image(img, mask, os.path.join(
                    out_dir, label + '_cam.jpg'))
            except Exception as e:
                print(f"  Warning: Layer {layer+1} failed - {str(e)}")
                continue
        
        print(f"✓ Completed {label}")
    
    print("\n✓ All visualizations completed successfully!")
    print(f"Results saved in: {sample_dir}")
