import keras
import numpy as np
import cv2
from Utils_model import VGG_LOSS

def change_model(model,new_shape=(None,384,384,3)):
    model._layers[0].batch_input_shape=new_shape
    new_model=keras.models.model_from_json(model.to_json())
    for layer in new_model._layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("loaded layer {}".format(layer.name))
        except:
            print("could not {}".format(layer.name))


    return new_model

def upscale():
    loss = VGG_LOSS((384,384,3))  
    model = keras.models.load_model( 'gen_model500.h5', custom_objects={'vgg_loss': loss.vgg_loss})
    # # print(model.summary())
    # inputs = keras.Input((384, 384, 3))

    # # Trace out the graph using the input:
    # outputs = model(inputs)

    # # Override the model:
    # # model = keras.model.Model(inputs, outputs)
    # model = keras.models.Model(inputs,outputs)
    # print(model.summary())
    img=cv2.imread("lr.jpg")
    img=(img.astype(np.float32) - 127.5)/127.5 
    output=model.predict(np.expand_dims(img, axis=0))
    # print(output)
    output=output[0]
    print(output.shape)

    output= (output + 1) * 127.5
    output= output.astype(np.uint8)
    # print(output)
    cv2.imwrite("sr.jpg",output)

# upscale()
loss = VGG_LOSS((384,384,3))  
model = keras.models.load_model( 'gen_model500.h5', custom_objects={'vgg_loss': loss.vgg_loss})
model=change_model(model,new_shape=(None,96,96,3))

# print(model.summary())

img=cv2.imread("lr.jpg")
img=(img.astype(np.float32) - 127.5)/127.5 
output=model.predict(np.expand_dims(img, axis=0))
print(output)
output=output[0]
print(output.shape)
output= (output + 1) * 127.5
output= output.astype(np.uint8)
print(output)
cv2.imwrite("sr_500.jpg",output)