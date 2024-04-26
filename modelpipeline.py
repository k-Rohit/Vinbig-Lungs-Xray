import streamlit as st


def pipeline():
    st.title("Model Pipeline")

    st.image('flowchart.jpeg',caption='Flowchart')

    st.subheader("Our pipeline consists of two models - A classifier and a detector")

    st.subheader("Classifier")

    st.markdown('''

- The DaViT_UnetR_Model combines the strengths of two architectures: DaViT (Dual Attention Vision Transformers) and U-Net.
- DaViT captures global context efficiently with self-attention mechanisms using spatial and channel tokens.
- U-Net specializes in biomedical image segmentation, featuring an encoder-decoder structure.
- The model begins with a pre-trained DaViT backbone to process input images and generate hierarchical feature representations.
- These representations encode both low-level details and high-level semantic information.
- An encoder-decoder architecture inspired by U-Net further processes these features.
- Encoders downsample feature maps to capture hierarchical features, while decoders upsample to recover spatial details.
- Skip connections facilitate information flow between layers.
- After decoding, convolutional layers further process features, increasing complexity.
- These layers bridge the encoder-decoder part with the classifier.
- The classifier consists of fully connected layers, ReLU activation functions, and dropout layers to prevent overfitting.
- The final output represents logits for each class, typically converted to probabilities using softmax during inference.
''')
    
    st.subheader("Detector")

    st.markdown('''

    - Anchor-Free Detection: YOLOv8 adopts an anchor-free approach, directly predicting the center of objects instead of relying on predefined anchor boxes. This simplifies the detection process and reduces the number of box predictions, leading to faster inference speeds.
 - New Convolutions: The model incorporates changes in its convolutional architecture, including replacing the stem's 6x6 convolution with a 3x3 convolution, modifying the main building block, and updating the C2f and C3 modules. These changes, such as adjusting kernel sizes and concatenating features, align YOLOv8 more closely with the ResNet block introduced in 2015.
- Mosaic Augmentation: YOLOv8 enhances its training routine with online image augmentation, including mosaic augmentation. However, unlike previous versions, mosaic augmentation is turned off for the last ten training epochs to prevent performance degradation. This attention to training methodology underscores the model's robustness and performance improvements.
 - Accuracy Improvements: YOLOv8's development is driven by empirical evaluation on the COCO benchmark, resulting in state-of-the-art accuracy at comparable inference speeds. The model undergoes continuous experimentation and validation to refine its components and achieve superior performance.



''')

def main():
    pipeline()

if __name__ == '__main__':
    pipeline()