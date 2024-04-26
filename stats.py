import streamlit as st
import pandas as pd

def show_statistics():




    st.title("Problem Statement")
    st.markdown('''

    In this competition, we need to  automatically localize and classify 14 types of thoracic abnormalities from chest radiographs. 
    The dataset consists of 18,000 scans that have been annotated by experienced radiologists. 
    For training the model 15,000 independently-labeled images are there
    and the model will be evaluated on a test set of 3,000 images. 
    These annotations were collected via VinBigData's web-based platform, VinLab. 
                
    This will act as a valuable second opinion for radiologists. 
    An automated system that could accurately identify and localize findings on chest radiographs would relieve the stress of busy doctors while also providing patients with a more accurate diagnosis

''')
    st.title("Dataset Description")
    st.markdown('''

The dataset comprises 18,000 postero-anterior (PA) CXR scans in DICOM format, which were de-identified to protect patient privacy.

Postero-anterior (PA) refers to a specific direction of X-ray imaging commonly used in chest X-rays (CXR). In a PA chest X-ray, the X-ray beam passes through the patient's body from the posterior (back) side to the anterior (front) side, with the detector placed on the anterior side to capture the resulting image.

All images were labeled by a panel of experienced radiologists for the presence of 14 critical radiographic findings as listed below:

0 - Aortic enlargement  
1 - Atelectasis  
2 - Calcification  
3 - Cardiomegaly  
4 - Consolidation  
5 - ILD  
6 - Infiltration  
7 - Lung Opacity  
8 - Nodule/Mass  
9 - Other lesion  
10 - Pleural effusion  
11 - Pleural thickening  
12 - Pneumothorax  
13 - Pulmonary fibrosis  
14 - "No finding"


''')
   
#     train_df = pd.read_csv("train_vinbig.csv")
#     x = train_df['class_name'].value_counts().keys()
#     y = train_df['class_name'].value_counts().values
#     count_dict = {'Class_Name' : x ,
#                 'Number of classes' : y}
#     count_df = pd.DataFrame(count_dict)
#     count_df = count_df.sort_values(by='Class_Name')

# # Display the bar chart with a title and sorted data in Streamlit
#     st.bar_chart(count_df.set_index('Class_Name'), use_container_width=True)

# # Add a title
#     st.title('Class Distribution')

def main():
    show_statistics()

if __name__ == "__main__":
    main()
