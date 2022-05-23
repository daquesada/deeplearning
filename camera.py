#!/usr/bin/env python
print("cargando")
import PySimpleGUI as sg
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import os

"""
Demo program that displays a webcam using OpenCV
"""
print("cargado")
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def main():

    sg.theme('Black')
    result_list = [
	      [sg.Text('Imagen', justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image'),],
              [sg.Text('',key="Resultado1",visible=False),sg.Text("",key="per1",visible=False)],
              [sg.Text('', key="Resultado2",visible=False), sg.Text("",key="per2",visible=False)], 
              [sg.Text('',key="Resultado3",visible=False),sg.Text("",key="per3",visible=False)],
    ]
    guia_list = [
	[sg.Image(filename='', key='image1',visible=False)],
        [sg.Text("", key="exa1",visible=False)],
        [sg.Image(filename="",key="image2",visible=False)],
        [sg.Text("",key="exa2",visible=False)],
        [sg.Image(filename="",key="image3",visible=False)],
        [sg.Text("",key="exa3",visible=False)]
    ]

    # define the window layout
    layout = [[
	sg.Column(result_list),
	sg.VSeparator(),
	sg.Column(guia_list, key="guia", visible=False),
	[sg.Button('Escanear', size=(10, 1), font='Helvetica 14'),
         sg.Button('Identificar', size=(10, 1), font='Any 14'),
         sg.Button('Salir', size=(10, 1), font='Helvetica 14'), ]
	],
        
    ]

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
            window['Resultado1'].update("",visible=False)
            window['Resultado2'].update("",visible=False)
            window['Resultado3'].update("",visible=False)
            window['per1'].update("",visible=False)
            window['per2'].update("",visible=False)
            window['per3'].update("",visible=False)
            window['image1'].update(filename="",visible=False)
            window['image2'].update(filename="",visible=False)
            window['image3'].update(filename="",visible=False)
            window['exa3'].update("",visible=False)
            window['exa2'].update("",visible=False)
            window['exa1'].update("",visible=False)
            window['guia'].update(visible=False)
            recording = True

        elif event == 'Identificar':
            recording = False
            cv2.imwrite("captura.png",image_cv)
            img_loaded = cv2.resize(image_cv,(200, 200))
            imgbytes = cv2.imencode('.png', img_loaded)[1].tobytes()
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
            maxs = df.loc[0].sort_values(ascending=False)
            maxs = round(maxs * 100,2 )
            idx = 1
            window['guia'].update(visible=True)
            for key, value in maxs[:3].items():

                window['Resultado{}'.format(idx)].update(value="{}".format(key),visible=True)
                window['per{}'.format(idx)].update("{}%".format(round(value,2)),visible=True)
                window['exa{}'.format(idx)].update(key,visible=True)
                img = cv2.imread(os.path.join("{}".format(key),"captura.jpg")) 
                resized = cv2.resize(img,(90,80))
                img_bytes = cv2.imencode('.png', resized)[1].tobytes()
                window['image{}'.format(idx)].update(data=img_bytes,visible=True)
                idx = idx + 1
        if recording:
            ret, frame = cap.read()
            img_output = cv2.resize(frame,(480,260))
            imgbytes = cv2.imencode('.png', img_output)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
            image_cv = frame
            # window['image'].update(filename="Captura2.png")



main()
