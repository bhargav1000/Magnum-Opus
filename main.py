#trained over 588 train images and 252 test images

from opclass import Classifier as oc
a=oc()
a.init_model()
b=a.get_prediction('demo_pan3.jpg')
print("Predicted:",b)