#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

"""
Demo program that displays a webcam using OpenCV
"""
print("loaded")
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def main():

    sg.theme('Black')

    # define the window layout
    layout = [[sg.Text('IMACA', size=(10, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Text('Resultado', size=(10, 1), justification='center', font='Helvetica 20'), 
              sg.Text('', size=(10, 1), justification='center', font='Helvetica 20',key="Resultado")],
              [sg.Button('Escanear', size=(10, 1), font='Helvetica 14'),
               sg.Button('Capturar', size=(10, 1), font='Any 14'),
               sg.Button('Salir', size=(10, 1), font='Helvetica 14'), ]]

    # create the window and show it without the plot
    window = sg.Window('IMACA',
                       layout, location=(0,0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    recording = False
    image_cv = np.full((640, 480), 255)
    while True:
        event, values = window.read(timeout=20)
        
        if event == 'Salir' or event == sg.WIN_CLOSED:
            return

        elif event == 'Escanear':
            window['Resultado'].update("")
            recording = True

        elif event == 'Capturar':
            recording = False
            cv2.imwrite("captura.png",image_cv)
            imgbytes = cv2.imencode('.png', image_cv)[1].tobytes()
            window['image'].update(data=imgbytes)
            img, x = get_image('captura.png')
            x= x.astype('float32') / 255.
            #Load model
            interpreter = tf.lite.Interpreter(model_path="resnet_model.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            #load image
            input_shape = input_details[0]['shape']
            input_data = x
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            f = open("labels.txt","r")
            labels = f.read()
            labels = labels.split("\n")
            labels = labels[:len(labels)-1]

            df = pd.DataFrame(data=output_data,columns=labels)

            print("{}".format(df.loc[0]))


            window['Resultado'].update("{}".format(df.loc[0]))

        if recording:
            ret, frame = cap.read()
            img_output = cv2.resize(frame,(200,200))
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
            image_cv = frame
            # window['image'].update(filename="Captura2.png")



main()
