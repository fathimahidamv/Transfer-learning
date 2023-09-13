import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
model= load_model('weight.h5')

# testing the model
# def testing_image(image_directory):
path=r"C:\Users\fathi\OneDrive\Desktop\DATASETS\flower_dataset\rose\7461897646_3203dbe067_n.jpg"
test_image = image.load_img(path, target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = model.predict(x= test_image)
print(result)
if np.argmax(result)  == 0:
  prediction = 'daisy'
elif np.argmax(result)  == 1:
  prediction = 'dandelion'
elif np.argmax(result)  == 2:
  prediction = 'rose'
elif np.argmax(result)  == 3:
  prediction = 'sunflower'

else:
  prediction = 'tulip'

print(prediction)

  

