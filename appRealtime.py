import cv2
import torch
import torchvision
import numpy as np
import os 

# pip3 install torch torchvision opencv-python
# https://towardsdatascience.com/semantic-image-segmentation-with-deeplabv3-pytorch-989319a9a4fb
# https://pytorch.org/vision/stable/models.html#semantic-segmentation

def load_model():
    # Load the DeepLab v3 model to system
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    #model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
    #model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    #model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    #model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

    model.to(device).eval()
    return model

def get_pred(img, model):
  # See if GPU is available and if yes, use it
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Define the standard transforms that need to be done at inference time
  imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
  preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                std  = imagenet_stats[1])])
  input_tensor = preprocess(img).unsqueeze(0)
  input_tensor = input_tensor.to(device)

  # Make the predictions for labels across the image
  with torch.no_grad():
      output = model(input_tensor)["out"][0]
      output = output.argmax(0)

  # Return the predictions
  return output.cpu().numpy()


if __name__ == "__main__":

    # Open video capture
    cap = cv2.VideoCapture(0)
    
    # Load Model
    model = load_model()

    # Background image
    bg_image = cv2.imread(os.path.join("images","backgrounds","arco-login.jpg"),1)
    #bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame,(256,256))
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width, height, channels = frame.shape
        labels = get_pred(frame, model)

        # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
        # Hence wherever person is predicted, the label returned will be 15
        # Subsequently repeat the mask across RGB channels 
        mask = labels == 15
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)

        # Resize the image as per the frame capture size
        bg = cv2.resize(bg_image, (height, width))
        bg[mask] = frame[mask]
        frame = bg

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



