import streamlit as st
import os
import cv2
import shutil
from PIL import Image
import torchvision.transforms as transforms
import torch
import timm
import torch.nn as nn
from ultralytics import YOLO
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from stats import show_statistics
from modelpipeline import pipeline
from streamlit_lottie import st_lottie 


# Ensure necessary folders exist
os.makedirs("temp", exist_ok=True)
os.makedirs("Disease_Present", exist_ok=True)
os.makedirs("Disease_Absent", exist_ok=True)
os.makedirs("result", exist_ok=True)
st.set_page_config(page_title="Lung's Abnormality Detection")

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


def classify_and_move(image_path, image_tensor):
    output_logits = model(image_tensor)
    predicted_label = torch.argmax(output_logits, dim=1)
    target_folder = "Disease_Absent" if predicted_label == 1 else "Disease_Present"
    shutil.move(image_path, os.path.join(target_folder, os.path.basename(image_path)))
    return target_folder
    

def print_output(folder_path):
    image_files = os.listdir(folder_path)
    disease_present_folder = "Disease_Present"
    disease_absent_folder = "Disease_Absent"
    
    if not os.path.exists(disease_present_folder):
        os.makedirs(disease_present_folder)
        
    if not os.path.exists(disease_absent_folder):
        os.makedirs(disease_absent_folder)
    
    for image in image_files:
        if image.endswith(('.png','.jpg','.jpeg')):
            image_path = os.path.join(folder_path, image)
            image_tensor = preprocess_image(image_path, image_size)
            image_tensor = image_tensor.to(device)
            output_logits = model(image_tensor)
            predicted_label = torch.argmax(output_logits, dim=1)
            if predicted_label == 1:
                shutil.move(image_path, disease_absent_folder)
            else:
                shutil.move(image_path, disease_present_folder)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Disease Classification", "Dataset Statistics","Model Pipeline"])
    if page == "Disease Classification":
        st.title("Lung's X-ray Abnormalities Detection")
        uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
        if uploaded_files:
            columns = {
                "Disease_Present": [],
                "Disease_Absent": []
            }

            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                file_path = f"temp/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(bytes_data)

                image_tensor = preprocess_image(file_path, IMAGE_SIZE).to(device)
                folder = classify_and_move(file_path, image_tensor)
                columns[folder].append(uploaded_file.name)

            col1, col2= st.columns(2)
            image_height = 300
            # with col1:
            #     st.header("Diseased")
            #     for image_name in columns["Disease_Present"]:
            #         st.image(f"Disease_Present/{image_name}", caption=image_name)

            with col1:
                st.header("Non-Diseased")
                for image_name in columns["Disease_Absent"]:
                    st.image(f"Disease_Absent/{image_name}", caption=image_name)
            with col2:
                st.header("Diseased")
                diseased_files = [os.path.join("Disease_Present", f) for f in os.listdir("Disease_Present")]
                # Initialize YOLO model
                model = YOLO('best_vinbig.pt')
                    
                # Process each image with YOLO and save the detection results
                results = model.predict(diseased_files,save=True)  # return a list of Results objects

                detection_output=os.listdir("runs/detect/predict")

                for file_name in detection_output:
                    st.image(os.path.join("runs/detect/predict",file_name), caption=file_name)

                
                shutil.rmtree('runs/detect/predict')

                # for i, result in enumerate(results):
                #         # Save the detection result image
                #     result.save(filename=f'result/result_{i}.jpg')
                # result_files = os.listdir("result")
                # for file_name in result_files:
                #     st.image(os.path.join("result", file_name), caption=file_name)



    elif page == "Dataset Statistics":
        show_statistics()
    elif page == "Model Pipeline":
        pipeline()

    st.sidebar.markdown(
        """
        ## Application Information
        This application is designed to classify diseases from medical images.
        You can navigate between different pages using the sidebar.
        """
    )


if __name__ == "__main__":
    main()



