# Magnum-Opus
A CNN based Image Classifier used to classify Aadhaar Card, PAN Card and any other document. This model was trained on a dataset of Aadhaar Cards, PAN cards and Other documents like gas bills, voter ID cards, driving licence etc. collected from customer data. The model was trained over several variations of the images such as blurred or tilted images. The model has an accuracy of 94%. 

## Requirements:
- keras 2.3.0
- cuda toolkit 9.0 (for GPU)

## Model Overview:
- The model consists of 2 parts: 
  1. A vgg16 pre-trained setup
  2. Fully Connected Neural Network
 
### Training: 
- Due to privacy reasons I was given a very limited dataset, so I had to upsample this data to ensure that I could train my model for all possible scenarios. I used the ```ImageDataGenerator``` module from keras and added variations to the dataset like tilting or selective blurring and increased my dataset size for each class.  
  
- The VGG16 model already contains pre-trained weights used to classify basic objects like mugs or fruits, but in my case I had to adjust these weights and use the architecture over my dataset so I froze the 4 end layers which were used to classify images, now these layers were basically the fully connected neural network layers of the vgg16 model used to classify images right after the flattened image was sent to this network. 
(You can learn more about vgg16 [here](https://neurohive.io/en/popular-networks/vgg16/).) 

- Freezing the 4 layers allowed me to utilize only the convoluted layers and send the output to a custom fully connected neural network which would be adjusted for only specific images i.e., Aadhaar Card or PAN Card. Given that I had a limited dataset I had to include dropout in the fully connected network to prevent overfitting on the data. 

- This process ensured that the model would understand only the specific document types required to be classified.

### Evaluation:
- The trained model was then evaluated on accuracy and through manual checking methods to ensure that the right cards/documents were being classified. The model has an accuracy of 94%.

- The operations team manually verified several images using the classifier to ensure that the right documents were classified.

## Getting started:
- Train the model. Please keep your dataset images in a directory called `train` with 2 sub-directories called `aadhar` and `pan` which will contain the respective images. Also create a directory called `validation` in a similar fashion. You can change the directory and their names, just edit **line37** and **line38** in [train2.py](https://github.com/bhargav1000/Magnum-Opus/blob/master/model/train2.py)
- Add the resulting model file in the `init_model` function using the `model` argument.
- Edit **line 6** in ```main.py``` and place your image path. 
- Run ```main.py``` and check the classification.

## Impact of the project:
- This project was packaged into a ready to use library which helped save more 50 hours of manual customer document verification. This helped the operations team to focus on other high priority tasks.

## Future Improvements:
- The model can be trained on fake documents and this can help in detecting fraudulent documents. 



