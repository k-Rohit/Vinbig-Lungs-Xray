import streamlit as st
import os
import shutil
from PIL import Image
import torchvision.transforms as transforms
import torch
import timm
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock

num_classes=2
BATCH_SIZE = 8
IMAGE_SIZE = (224,224)

class DaViT_UnetR_Modelv2(nn.Module):
    def __init__(self, num_classes, pretrained=True, fine_tune=False):
        super(DaViT_UnetR_Modelv2, self).__init__()
        
        self.davit = timm.create_model('davit_base.msft_in1k', pretrained=pretrained, features_only=True, in_chans=1)
        
        if not fine_tune:
            for param in self.davit.parameters():
                param.requires_grad = False
        
        
        spatial_dims = 2 
        in_channels = 1 # R,G,B
        feature_size = 128
        norm_name = "instance"
        hidden_size = 128
        res_block = True
        conv_block = False

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=1,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size*2,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=1,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size*4,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=1,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size * 8,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(feature_size, 78, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(78, 50, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layer with convolution
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2450, 1024),  # (DYNAMIC)Adjust the input size based on the output size of the convolutional layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    
    def forward(self, x_in):
        
        hidden_states_out = self.davit(x_in) # returns 4 lists
#         print("Length of hidden states from DaViT:", len(hidden_states_out))
#         for i in hidden_states_out:
#             print(i.shape)
#         print()


        enc1 = self.encoder1(x_in)
#         print("output from encoder1:", enc1.shape)
        
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(x2)
#         print("output from encoder2:", enc2.shape)
        
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(x3)
#         print("output from encoder3:", enc3.shape)
        
        
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(x4)
#         print("output from encoder4:", enc4.shape)
        
#         print("All encoders OK\n")
        
        dec4 = hidden_states_out[3]
#         print("Input to decoder5:", dec4.shape, enc4.shape)
        dec3 = self.decoder5(dec4, enc4)
#         print("output from decoder5:", dec3.shape)
        
#         print("Input to decoder4:", dec3.shape, enc3.shape)
        dec2 = self.decoder4(dec3, enc3)
#         print("output from decoder4:", dec2.shape)
        
#         print("Input to decoder3:", dec2.shape, enc2.shape)
        dec1 = self.decoder3(dec2, enc2)
#         print("output from decoder3:", dec1.shape)
        
#         print("Input to decoder2:", dec1.shape, enc1.shape)
        out = self.decoder2(dec1, enc1) 
#         print("output from decoder2:", out.shape)
        

        
        conv_out = self.conv(out)
#         print(f"conv_out_shape:{conv_out.shape}")

        return self.classifier(conv_out)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DaViT_UnetR_Modelv2(num_classes, fine_tune=False)
model.to(device)

# # print(model)
# print()

# x = torch.randn(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)


# output = model(x)
# print("Model output's shape:", output.shape)
# print(output) # logits 

model.load_state_dict(torch.load('classifier_model.pth', map_location=torch.device('cpu')))

model.eval() 

image_size = (224, 224)

def preprocess_image(image_path, image_size):
    
    # Load the image using PIL
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    # Resize the image to the required size
    image = image.resize(image_size)
    # Convert PIL Image to PyTorch tensor
    image_tensor = transforms.ToTensor()(image)
    # Normalize the image (if required)
    # You may need to adjust the mean and standard deviation values
    image_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(image_tensor)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def print_output(folder_path):
    
    image_files=os.listdir(folder_path)
    
    disease_present_folder="Disease_Present"
    disease_absent_folder="Disease Absent"
    
    if not os.path.exists(disease_present_folder):
        os.makedirs(disease_present_folder)
        
    if not os.path.exists(disease_absent_folder):
        os.makedirs(disease_absent_folder)
    
    for image in image_files:
        if image.endswith(('.png','.jpg','.jpeg')):
            image_path=os.path.join(folder_path,image)
            image_tensor=preprocess_image(image_path, image_size)
            image_tensor=image_tensor.to(device)
            output_logits=model(image_tensor)
            predicted_label=torch.argmax(output_logits, dim=1)
            if(predicted_label==1):
                shutil.copy(image_path,"Disease_Absent")
            else:
                shutil.copy(image_path,"Disease_Present")

def list_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    return files

def main():
    st.title("Folder Input Example")
    