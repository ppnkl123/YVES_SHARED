from flask import Flask, session, request, redirect, url_for, Response, json, render_template
import os, base64, cv2, random, string, csv, calendar, time
import numpy as np
from datetime import datetime
from pathlib import Path

##############################################################
##############################################################
# FIRST MODEL
import tensorflow as tf
from Test import detect_fn
from DetectModelTwo import detect_fn_roundTwo
from DetectNewModel import detect_fn_new

Path("./questionnaire/").mkdir(parents=True, exist_ok=True)
Path("./processed/").mkdir(parents=True, exist_ok=True)

path = os.path.join(os.getcwd(), "questionnaire")
nr = random.randint(0, 9999)
csvFile = os.path.join(path, "questionnaire" + str(nr) + ".csv")

app = Flask(__name__)

category_index = [{'name':'car', 'id':1}, {'name':'minion', 'id':2}, {'name':'rack', 'id':3}, {'name':'mixer', 'id':4}, {'name':'sandwich', 'id':5}]

category_index_sven_model = [{'name':'car', 'id':1}, {'name':'lion', 'id':2}, {'name':'elephant', 'id':3}]

def sven_model_detect():
    img = cv2.imread(os.path.join('processed', 'temp_img.jpg'))
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn_new(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    scores = detections['detection_scores']
    detection = detections['detection_classes']

        
    if len(detection) > 0:
        score = scores[0]
        detecteddd = detection[0]
        print(score, detecteddd)

        print("category index len", len(category_index_sven_model))
        print("det + label ", detection[0] + label_id_offset)

        if len(category_index_sven_model) > detection[0] + label_id_offset:
            labelobj =  category_index_sven_model[detection[0]]
            label = labelobj['name']
            print("")
            print("")
            print("sven model:  ")
            print("")
            print("")
            print(score, detecteddd)
            print(labelobj, label)
            print("FIRST DETECTION CLASS: ", detection[0])
            print("")
            print("")
            #label = labelobj['name']
            
            boxes = detections['detection_boxes']
            box = boxes[0]
            y1 = box[0] * 450 # in prozent vom gesamtbild?
            x1 = box[1] * 450
            y2 = box[2] * 450
            x2 = box[3] * 450

        else:
            label = "nothing"
            score = 0
            y1 = 0
            y2 = 0
            x1 = 0
            x2 = 0
        
        return(str(label), str(score), int(y1), int(x1), int(y2), int(x2))

    else:
        return("no detection")



def getDetection(image_np):
    if (type(image_np) == type(None)):
        img = cv2.imread(os.path.join('processed', 'temp_img.jpg'))
        image_np = np.array(img)

    print("")
    print("IMAGE SHAPE: ", image_np.shape)
    print("")
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    scores = detections['detection_scores']
    detection = detections['detection_classes']

        
    if len(detection) > 0:
        score = scores[0]
        detecteddd = detection[0]
        print(score, detecteddd)

        if len(category_index) > detection[0] + label_id_offset:
            labelobj =  category_index[detection[0]]
            label = labelobj['name']
            print("")
            print("")
            print(score, detecteddd)
            print(labelobj, label)
            print("FIRST DETECTION CLASS: ", detection[0])
            print("")
            print("")
            #label = labelobj['name']
            
            boxes = detections['detection_boxes']
            box = boxes[0]
            y1 = box[0] * 450 # in prozent vom gesamtbild?
            x1 = box[1] * 450
            y2 = box[2] * 450
            x2 = box[3] * 450

        else:
            label = "nothing"
            score = 0
            y1 = 0
            y2 = 0
            x1 = 0
            x2 = 0
        
        return(str(label), str(score), int(y1), int(x1), int(y2), int(x2))

    else:
        return("no detection")

##############################################################
##############################################################
# SECOND MODEL

def detectRoundTwo():
    #ret, frame = cap.read()
    img = cv2.imread(os.path.join('processed', 'temp_img.jpg'))
    image_np = np.array(img)
    print("")
    print("IMAGE SHAPE: ", image_np.shape)
    print("")
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn_roundTwo(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    scores = detections['detection_scores']
    detection = detections['detection_classes']

        
    if len(detection) > 0:
        score = scores[0]
        detecteddd = detection[0]

        if len(category_index) > detection[0] + label_id_offset:
            labelobj =  category_index[detection[0]]
            label = labelobj['name']
            print("")
            print("")
            print(score, detecteddd)
            print(labelobj, label)
            print("FIRST DETECTION CLASS: ", detection[0])
            print("")
            print("")
            #label = labelobj['name']
            
            boxes = detections['detection_boxes']
            box = boxes[0]
            y1 = box[0] * 450 # in prozent vom gesamtbild?
            x1 = box[1] * 450
            y2 = box[2] * 450
            x2 = box[3] * 450

        else:
            label = "nothing"
            score = 0
            y1 = 0
            y2 = 0
            x1 = 0
            x2 = 0
        
        return(str(label), str(score), int(y1), int(x1), int(y2), int(x2))

    else:
        return("no detection")



##############################################################
##############################################################

# Load model and allocate tensors for custom detection round 1
"""
interpreter1 = tflite.Interpreter(model_path='detect_car_1.tflite')
# allocate tensors
interpreter1.allocate_tensors()
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()

# Round 2

interpreter2 = tflite.Interpreter(model_path='detect_car_2.tflite')
# allocate tensors
interpreter2.allocate_tensors()
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

category_index = {
0 :'car',
1:'sandwich',
2: 'mixer'}

def customDetection():

    model_type = session.get('custom_model_type', None)

    image_path='processed/temp_img.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img,(320,320))

    input_tensor = np.array(np.expand_dims(img,0), np.float32)

    if model_type == '1':
        print("USING CUSTOM MODEL 1")
        input_index = interpreter1.get_input_details()[0]["index"]
        # input_index = input_details[0]['index']
        interpreter1.set_tensor(input_index, input_tensor)
        interpreter1.invoke()
        output_details = interpreter1.get_output_details()
        output_data = interpreter1.get_tensor(output_details[0]['index'])
    else:
        print("USING CUSTOM MODEL 2")
        input_index = interpreter2.get_input_details()[0]["index"]
        # input_index = input_details[0]['index']
        interpreter2.set_tensor(input_index, input_tensor)
        interpreter2.invoke()
        output_details = interpreter2.get_output_details()
        output_data = interpreter2.get_tensor(output_details[0]['index'])
    

    pred = np.squeeze(output_data)

    #boxes = np.squeeze(output_data['detection_boxes'])

    highest_pred_loc = np.argmax(pred)

    class_name = category_index[highest_pred_loc]
    message = {'name':class_name, 'accuracy':str(pred[0])}
    return message


#_______________________________________________________________
# LOADING IN YOLO
#
# Load Yolo

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# Get the last output layer to get the detection of the object
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Organise detections in arrays
boxes = []
confidences = []
class_ids = []
detections = []

responseDict = {"name":"test"}



#_______________________________________________________________
# DETECTING OBJECTS
#

def detect_obj(image):
    detection = detect_stuff(image)
    
    return detection

def detect_stuff(img):
    # Detecting Objects
    # Converting to Blob

        width = img.shape[1]
        height = img.shape[0]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 :
                        # object detected
                        # position of detection
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2])
                        print(w)
                        h = int(detection[3])
                        

                        x = int(center_x - w / 2)
                        y = int(center_y - h /2)

                        boxes.append([x, y, w, h])                 
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
                        x, y, w, h = boxes[-1]
                        label = classes[class_ids[-1]]
                        #confidence = classes[class_ids[-1]]
                        
                        return(label, str(confidence), x, y, w, h)
"""        

#_______________________________________________________________
# HANDLING ALL CLIENT SERVER COMMUNICATIONS
#
"""
@app.route('/getinformation', methods=['GET', 'POST'])
def getinformation():
    msgRECEIVE = request.json # get data from json
    msgRETURN = {}

    msg_type = msgRECEIVE["type"]
    msg_data = msgRECEIVE["data"]

    if msg_type == "save":
        # TODO
        print("SAVE IMAGE CAPTURE")
        
    
    elif msg_type == "detect_from_image":

        
        img_url = msg_data["image"]
        image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
        image_np = image64_to_numpy(image) # get numpy array representation of image 


        detection = detect_obj(image_np)
        if detection != None:
            label = detection[0]
            confidence = detection[1]
            x = detection[2]
            y = detection[3]
            #w = detection[4]
            #h = detection[5]
            object_label = session.get('user_label', None)
            #old_detection = session.get('detection', None)


            msgRETURN = {"type":"detection", "data":label, "confidence":confidence, "x":x, "y":y, "userInput":object_label}
            print("DETECTION FROM IMAGE")

    elif msg_type == "detect_cropped_image":

        print("MESSAGE DATA RECEIVED")
        if "image" in msg_data:
            img_url = msg_data["image"]
            image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
            image_np = image64_to_numpy(image) # get numpy array representation of image

            new_detection = detect_obj(image_np)
            
            if new_detection != None:
                label = new_detection[0]
                confidence = new_detection[1]
                x = new_detection[2]
                y = new_detection[3]
                #w = detection[4]
                #h = detection[5]
                object_label = session.get('user_label', None)


                msgRETURN = {"type":"detection", "data":label, "confidence":confidence, "x":x, "y":y, "userInput":object_label}
                print("DETECTION FROM CROPPED IMAGE")

    
    elif msg_type == "calculate":

        calc_type = msg_data["type"]
        fileName = msg_data["fileName"]
        imagePath = create_image_path(fileName)
        session[fileName] = imagePath

        if calc_type == "get_more_images":
            print("CALCULATING DIVERSIFIED IMAGES")
            images = diversify_images(imagePath)
            print("DIVERSIFIED IMAGES RESULT")
            print(str(images))
            msgRETURN = {"type":"diversified_images", "data":images}
        
        elif calc_type == "positions":
            print("CALCULATING DIVERSIFIED IMAGES")
            images = diversify_images(imagePath)
            print("DIVERSIFIED IMAGES RESULT")
            print(str(images))
            msgRETURN = {"type":"diversified_images_positions", "data":images}
        
            
   
    else:
        msgRETURN = {"type":"empty", "data":"empty"}
        print("MESSAGE RECEIVED BUT NOT READ")

    return Response(json.dumps(msgRETURN))
"""

#_______________________________________________________________
# REROUTING
#
@app.route('/<id>')
def reroute(id):

    if id == "restart":
        print("RESTARTING INTERACTION")
        return redirect(url_for('restart'))
    else:
        language = session.get('language', None)
        print("page ID:" + id)
        page = id + ".html"
        print("PAGE redirect:" + page)
        print("PAGE language:" + language)

        return render_template(page, language=language)

@app.route('/home')
def restart():
    number = generate_interaction()
    session['interaction_type'] = number
    generate_user_id()
    interaction_start()
    save_to_session('custom_model_type', '1')
    session.pop('og_det', None)
    session.pop('og_x', None)
    session.pop('og_y', None)
    
    return render_template("home.html")

#_______________________________________________________________
# HOME PAGE - CHOSE A LANGUAGE
#
@app.route('/home', methods=['POST', 'GET'])
def home():

    session['user_label'] = "?"
    number = generate_interaction()
    session['interaction_type'] = number
    session.pop('og_det', None)
    session.pop('og_x', None)
    session.pop('og_y', None)
    generate_user_id()
    save_to_session('csv', csvFile)
    interaction_start()
    save_to_session('custom_model_type', '1')

    if request.method == "POST":
        form_variable_input = request.form['language']

        if form_variable_input == 'en':
            language = "en"     # TODO
        elif form_variable_input == 'de':
            language = "de"     # TODO
        elif form_variable_input == 'rus':
            language = "rus"     # TODO
        elif form_variable_input == 'pol':
            language = "pol"     # TODO
        else:
            return print("couldn't figure out language")

        session['language'] = language
        page = "Introduction"
        print("PAGE TO REDIRECT TO: " + page)
        return redirect(url_for('reroute', id=page))
    else:
        return render_template('home.html')

@app.route('/launch', methods=['POST', 'GET'])
def launch():
    return render_template('launch.html')

#_______________________________________________________________
# 01 - 01 - CAN YOU TEACH ME SOMETHING?        -> put into body and take a picture
#


#_______________________________________________________________
# 01 - 02 - WHAT IS IT?        -> label the object
#

@app.route('/handle_form', methods=['POST'])
def handle_form():
    label = request.form['form_label']
    session['user_label'] = label
    print("")
    print("USER INPUT HERE")
    print("")
    print("label")
    print("")

    interaction = session.get('interaction_type', None)
    print("INTERACTION TYPE:" + str(interaction))
    
    if interaction == 0:
        print("interaction 0 - one step interaction")
        page_id = "21_00"
    else:
        print("Interaction 1 - two step interaction")
        page_id = "21"

    return redirect(url_for('reroute', id=page_id))




#_______________________________________________________________
# 01 - 03 - TEST ME        -> detect the object in restricted area
#


#_______________________________________________________________
# 01 - 04 - OOPS, CAN'T DETECT IT ALWAYS - NEED MORE IMAGES & CAN CREATE MORE IMAGES       -> diversify image
#

def diversify_images(imagePath):
    
    images_dict = []
    # TODO: Get ALL the altered images of the capture
    print("GET ALL IMAGES IN ALTERED VERSIONS")
    
    path = session.get('files_path', None)
    print("STATIC FILE PATH: ")
    print(path)
    print("IMAGE FILE PATH: ")
    print(imagePath)
    

    src = cv2.imread(imagePath) # as numpy

    #____________________________________________________________________
    # IMAGE 1 -  mirror at y
    images_dict.append(flip_at_x(src, path))
    
    # IMAGE 2 - mirror at x
    images_dict.append(flip_at_y(src, path))


    # IMAGE 3 - rotate 90°
    images_dict.append(rotate_90(src, path))


    # IMAGE 4 - rotate 180°
    images_dict.append(zoom_x(120, src, path))

    # IMAGE 5 - zoom in by 2.0
    images_dict.append(zoom(src, path))

    # ORIGINAL IMAGE
    images_dict.append(zoom_x(120, src, path))
    

    return images_dict

#_______________________________________________________________
# INTERACTION 2 - GENERATE IMAGES AT DIFFERENT POSITIONS
#
def diversify_images_position(imagePath):
    
    images_dict = []

    path = session.get('files_path', None)
    print("STATIC FILE PATH: ")
    print(path)
    print("IMAGE FILE PATH: ")
    print(imagePath)
    

    src = cv2.imread(imagePath) # as numpy

    cropped_im = crop_image(src, path) #TODO



    images_dict_temp = insert_crop(cropped_im, path) # returns an image array as json string, of original object cropped and pasted to different new positions
    images_dict = random.sample(images_dict_temp, k=len(images_dict_temp))

    images_dict.insert(0, original_image(src, path))

    print(len(images_dict))
    
    return images_dict

def diversify_images_rotation_and_position(imagePath):
    
    images_dict = []

    path = session.get('files_path', None)
    print("STATIC FILE PATH: ")
    print(path)
    print("IMAGE FILE PATH: ")
    print(imagePath)
    

    src = cv2.imread(imagePath) # as numpy

    cropped_im = crop_image(src, path) #TODO



    images_dict_pos = insert_crop(cropped_im, path) # returns an image array as json string, of original object cropped and pasted to different new positions
    images_dict_rot = rotate_image(cropped_im, path) # returns an image array as json string, of original object cropped and pasted to different new positions
    images_dict_temp = images_dict_pos + images_dict_rot
    images_dict = random.sample(images_dict_temp, k=len(images_dict_temp))

    images_dict.insert(0, original_image(src, path))

    print(len(images_dict))
    
    return images_dict

    
def crop_image(src, path):
    print(src.shape)
    cropped = src[60:390, 110:340]

    croppedFileName = "cropped.jpg"
    newImagePath = os.path.join(path, croppedFileName)
    cv2.imwrite(newImagePath, cropped)

    return(cropped)


# def crop_detected_object(src, path):
#     detection = getDetection("IMG")
#     y1 = detection[2]
#     x1 = detection[3]
#     y2 = detection[4]
#     x2 = detection[5]
#     print("")
#     print("coordinates of the image")
#     print("")
#     print(y1, x1, y2, x2)
#     print("")

#     #92 105 399 304

#     cropped_im = src[y1:y2, x1:x2]
#     print("SOURCE IMAGE: ", src.shape)
#     print("CROPPED IMAGE: ", cropped_im.shape)
#     croppedFile = "cropped_deprecated.jpg"
#     newImagePath = os.path.join(path, croppedFile)
#     cv2.imwrite(newImagePath, cropped_im)
#     return (cropped_im)

def insert_crop(cropped_im, path):

    x_offset = 0
    new_positions_image_array_jsn = []
 
    while x_offset <= 220:

        y_offset = 0
        
        while y_offset <= 120:

            newImage = "cropped_inserted_" + str(x_offset) + str(y_offset) + ".jpg"
            newImagePath = os.path.join(path, newImage)

            base_img = np.zeros([450,450,3],dtype=np.uint8)
            base_img.fill(255) # or img[:] = 255, make background white
            
            print("insert_crop",y_offset, x_offset, cropped_im.shape)

            if x_offset+cropped_im.shape[0] <=450 and y_offset+cropped_im.shape[1]<=450:
                base_img[x_offset:x_offset+cropped_im.shape[0], y_offset:y_offset+cropped_im.shape[1]] = cropped_im
                cv2.imwrite(newImagePath, base_img)

                with open(newImagePath, "rb") as image_file:
                    crop_inserted_enc_str = base64.b64encode(image_file.read())

                new_positions_image_array_jsn.append(str(crop_inserted_enc_str).replace("b\'", "").replace("\'", ''))

            y_offset += 20

        x_offset += 20
    
    return new_positions_image_array_jsn

def rotate_image(cropped_im, path):

    images = 0
    angle_degree = 17
    new_rotations_image_array_jsn = []
 
    while images < 20:

            h, w = cropped_im.shape[:2]
            center = (w/2, h/2)

            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle_degree, scale=1)

            rotated_image = cv2.warpAffine(src=cropped_im, M=rotate_matrix, dsize=(300, 300), borderValue=(255,255,255))


            rotatedFile = "rotate" + str(angle_degree) + ".jpg"
            tempImagePath = os.path.join(path, rotatedFile)

            cv2.imwrite(tempImagePath, rotated_image)
            with open(tempImagePath, "rb") as image_file:
                crop_inserted_enc_str = base64.b64encode(image_file.read())

            new_rotations_image_array_jsn.append(str(crop_inserted_enc_str).replace("b\'", "").replace("\'", ''))

            angle_degree += 17
            images += 1
    
    return new_rotations_image_array_jsn



def diversify_images_rotation(imagePath):
        
    images_dict = []
    # TODO: Get ALL the altered images of the capture
    print("GET ALL IMAGES IN ALTERED VERSIONS")
    
    path = session.get('files_path', None)
    print("STATIC FILE PATH: " + path)
    print("IMAGE FILE PATH: " + imagePath)

    src = cv2.imread(imagePath) # as numpy

    cropped_im = crop_image(src, path) #TODO

    images_dict_temp = rotate_image(cropped_im, path) # returns an image array as json string, of original object cropped and pasted to different new positions
    images_dict = random.sample(images_dict_temp, k=len(images_dict_temp))

    images_dict.insert(0, original_image(src, path))

    print(len(images_dict))

    return images_dict

def diversify_images_zoom(imagePath):
    images_dict = []
    # TODO: Get ALL the altered images of the capture
    print("GET ALL IMAGES IN ALTERED VERSIONS")
    
    path = session.get('files_path', None)
    print("STATIC FILE PATH: " + path)
    print("IMAGE FILE PATH: " + imagePath)
    

    src = cv2.imread(imagePath) # as numpy

    #____________________________________________________________________
    # IMAGE 1 -  mirror at y
    images_dict.append(zoom_x(120, src, path))

    # IMAGE 2 - rotate 90°
    images_dict.append(zoom_x(100, src, path))

    # IMAGE 3 - mirror at x
    images_dict.append(zoom_x(80, src, path))

    # IMAGE 4 - mirror at x
    images_dict.append(zoom_x(50, src, path))

    return images_dict


#
# ZOOM
#
def zoom_x(zoomNr, image, path):
    strx = session.get('og_x', None)
    print("STRX: ", str(strx))
    stry = session.get('og_y', None)
    print("STRX: ", str(stry))

    if strx == "" or strx == None:
        strx = 225
    if stry == "" or stry == None:
        stry = 225
 
    x = int(float(strx))
    y = int(float(stry))

    x1 = x-zoomNr
    x2 = x+zoomNr

    y1 = y-zoomNr
    y2 = y+zoomNr

    if (x1 <0):
        x1 = 0
    
    if (x2 >450):
        x2 = 450
    
    if (y1 <0):
        y1 = 0
    
    if (y2 >450):
        y2 = 450

    print("COORDINATES: ", x, y, x1, x2, y1, y2)


    cropped_im = image[y1:y2, x1:x2]

    zoomedFile = "zoomed" + str(zoomNr) + ".jpg"
    newImagePath = os.path.join(path, zoomedFile)

    cv2.imwrite(newImagePath, cropped_im)
    with open(newImagePath, "rb") as image_file:
        zoomed_encoded_string = base64.b64encode(image_file.read())
    
    return str(zoomed_encoded_string).replace("b\'", "").replace("\'", '')


#
# ORIGINAL IMAGE
#
def original_image(src, path):
    # create flipped image file & path
    fileName = "temp_pos.jpg"
    filePath = os.path.join(path, fileName)

    # save
    cv2.imwrite(filePath, src)
    with open(filePath, "rb") as image_file:
        new = base64.b64encode(image_file.read())
    
    return str(new).replace("b\'", "").replace("\'", '')




#_______________________________________________________________
# FLIP IMAGE
#
def flip_at_x(image, path):
    # create flipped image file & path
    flippedFile = "flippedX.jpg"
    imagePathFlippedX = os.path.join(path, flippedFile)

    # flip image
    flippedImageX = cv2.flip(image, 0)

    # save
    cv2.imwrite(imagePathFlippedX, flippedImageX)
    with open(imagePathFlippedX, "rb") as image_file:
        flipped_image_x_encoded_string = base64.b64encode(image_file.read())
    
    return str(flipped_image_x_encoded_string).replace("b\'", "").replace("\'", '')

def flip_at_y(image, path):
    # create flipped image file
    flippedFileY = "flippedY.jpg"
    imagePathFlippedY = os.path.join(path, flippedFileY)
    
    flippedImageY = cv2.flip(image, 1)
    
    # save
    cv2.imwrite(imagePathFlippedY, flippedImageY)
    with open(imagePathFlippedY, "rb") as image_file:
        flipped_image_y_encoded_string = base64.b64encode(image_file.read())

    return str(flipped_image_y_encoded_string).replace("b\'", "").replace("\'", '')


def rotate_90(image, path):
    rotate90File = "rotate90.jpg"
    imagePathRotate90 = os.path.join(path, rotate90File)
    rotate90Image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite(imagePathRotate90, rotate90Image)
    with open(imagePathRotate90, "rb") as image_file:
        rotate_90_image_encoded_string = base64.b64encode(image_file.read())

    return str(rotate_90_image_encoded_string).replace("b\'", "").replace("\'", '')

def rotate_180(image, path):
    rotatedFile = "rotate180.jpg"
    imagePathRotate180 = os.path.join(path, rotatedFile)
    rotate180Image = cv2.rotate(image, cv2.ROTATE_180)

    cv2.imwrite(imagePathRotate180, rotate180Image)
    with open(imagePathRotate180, "rb") as image_file:
        rotated_180_encoded_string = base64.b64encode(image_file.read())
    
    return str(rotated_180_encoded_string).replace("b\'", "").replace("\'", '')

#____________________________________________________________________________

def rotate_45(image, path):

    height, width = image.shape[:2]
    center = (width/2, height/2)

    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=(255,255,255))


    rotatedFile = "rotate45.jpg"
    imagePathRotate45 = os.path.join(path, rotatedFile)

    cv2.imwrite(imagePathRotate45, rotated_image)
    with open(imagePathRotate45, "rb") as image_file:
        rotated_45_encoded_string = base64.b64encode(image_file.read())
    
    return str(rotated_45_encoded_string).replace("b\'", "").replace("\'", '')


#____________________________________________________________________________

def rotate_200(image, path):

    height, width = image.shape[:2]
    center = (width/2, height/2)

    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=200, scale=1)

    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=(255,255,255))


    rotatedFile = "rotate200.jpg"
    imagePathRotate200 = os.path.join(path, rotatedFile)

    cv2.imwrite(imagePathRotate200, rotated_image)
    with open(imagePathRotate200, "rb") as image_file:
        rotated_200_encoded_string = base64.b64encode(image_file.read())
    
    return str(rotated_200_encoded_string).replace("b\'", "").replace("\'", '')

def rotate_120(image, path):

    height, width = image.shape[:2]
    center = (width/2, height/2)

    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=120, scale=1)

    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=(255,255,255))


    rotatedFile = "rotate120.jpg"
    imagePathRotate120 = os.path.join(path, rotatedFile)

    cv2.imwrite(imagePathRotate120, rotated_image)
    with open(imagePathRotate120, "rb") as image_file:
        rotated_120_encoded_string = base64.b64encode(image_file.read())
    
    return str(rotated_120_encoded_string).replace("b\'", "").replace("\'", '')




def zoom(image, path):
    cropped_im = image[50:300, 50:300]

    zoomedFile = "zoomed.jpg"
    newImagePath = os.path.join(path, zoomedFile)

    cv2.imwrite(newImagePath, cropped_im)
    with open(newImagePath, "rb") as image_file:
        zoomed_encoded_string = base64.b64encode(image_file.read())
    
    return str(zoomed_encoded_string).replace("b\'", "").replace("\'", '')



#_______________________________________________________________
# 01 - 05 - HERE ARE THE RESULTING IMAGES - NOW TRAIN       -> trigger training
#
@app.route('/handle_training', methods=['POST'])
def handle_training():
    
    page_id = "13"
    save_to_session('custom_model_type', '2')

    return redirect(url_for('reroute', id=page_id))

#_______________________________________________________________
# 01 - 06 - TEST ME AGAIN       -> now detect always
#

################################################################
#_______________________________________________________________
# INTERACTION 2
#
#_______________________________________________________________
# POSITION
#
#@app.route('/21')
#def p21():
#    return nextpage(url_for('nextpage', page="21-en.html"))


#@app.route('/<page>')
#def nextpage(page):
#    return render_template(page)

#_______________________________________________________________
# HANDLE FORMS
#
@app.route('/handle_forms', methods=['POST'])
def handle_forms():
    answer = request.form['answer']

    #if answer == "26":
    #   save_to_session('custom_model_type', '2')

    return redirect(url_for('reroute', id=answer))


#_______________________________________________________________
# HANDLING ALL CLIENT SERVER COMMUNICATIONS
#
@app.route('/testFetch', methods=['GET', 'POST'])
def testFetch():
    msgRECEIVE = request.json # get data from json
    requestType = msgRECEIVE["requestType"]
    requestMsg = msgRECEIVE["requestMsg"]

    print("REQUEST RECEIVED: " + requestType)

    if requestType == "detect":
        if requestMsg != "":

            img_url = requestMsg["image"]
            image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
            image_np = image64_to_numpy(image) # get numpy array representation of image 
            save_image(image_np, "temp_img.jpg")

            detection = getDetection(image_np)
            print("  ")
            print("GIMME GIMME GIMME A RESULT: ", detection)
            print("  ")

            name = detection[0]
            confidence = detection[1]
            y1 = detection[2]
            x1 = detection[3]
            y2 = detection[4]
            x2 = detection[5]

            w = x2 - x1
            h = y2 - y1

            x = int(x1 + w/2)
            y = int(y1 + h/2)

            print("  ")
            print("GIMME GIMME GIMME AN INTEGER: ", x, y)
            print("  ")

            save_to_session('last_det', name)
            save_to_session('last_x', x)
            save_to_session('last_y', y)

            object_label = session.get('user_label', None)
            language = session.get('language', None)

            # Sven: This seems to be odd. It's the same image, why should the detection be different?
            firstDet = session.get('og_det')
            if (firstDet == None):
                detectionNew = getDetection(image_np)
                firstDet = detectionNew[0]
                save_to_session('og_det', firstDet)

            responseMessage = {"custom_detection_name":name, "custom_detection_confidence":confidence, "x":x, "y":y,"x_old":session.get('og_x'), "y_old":session.get('og_y'), "userInput":object_label, "language":language, "firstDet":firstDet, "rectangle":[y1,x1,y2,x2]} 
            responseType = "detection"
            
        
            """
            img_url = requestMsg["image"]
            image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
            image_np = image64_to_numpy(image) # get numpy array representation of image

            custom_detection = customDetection()

            custom_detection_name = custom_detection['name']
            custom_detection_confidence = custom_detection['accuracy']
            print("  ")
            print("CUSTOM DETECTION: ", custom_detection_name, custom_detection_confidence)
            print("  ")

            new_detection = detect_obj(image_np)
            print("NEW DETECTION", new_detection)

            object_label = session.get('user_label', None)
            language = session.get('language', None)
                 
            
            if new_detection != None:
                yolo_label = new_detection[0]
                yolo_confidence = new_detection[1]
                x = new_detection[2]
                y = new_detection[3]

                save_to_session('last_det', yolo_label)
                save_to_session('last_x', x)
                save_to_session('last_y', y)

                print("OG DETECTION", session.get('og_x', None), session.get('og_y', None), session.get('og_det', None))

                responseMessage = {"label":yolo_label, "confidence":yolo_confidence, "custom_detection_name":custom_detection_name, "custom_detection_confidence":custom_detection_confidence, "x":x, "y":y,"x_old":session.get('og_x'), "y_old":session.get('og_y'), "userInput":object_label, "language":language, "firstDet":session.get('og_det')} 
                responseType = "detection"
                

            else:
                responseMessage = {"custom_detection_name":custom_detection_name, "custom_detection_confidence":custom_detection_confidence, "userInput":object_label, "x":180, "y":200} 
                responseType = "detection"
            """
    elif requestType == "detect2":
        if requestMsg != "":

            img_url = requestMsg["image"]
            image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
            image_np = image64_to_numpy(image) # get numpy array representation of image 
            save_image(image_np, "temp_img.jpg")

            detection = detectRoundTwo()
            print("  ")
            print("GIMME GIMME GIMME A RESULT ROUND TWO: ", detection)
            print("  ")

            name = detection[0]
            confidence = detection[1]
            y1 = detection[2]
            x1 = detection[3]
            y2 = detection[4]
            x2 = detection[5]

            w = x2 - x1
            h = y2 - y1

            x = int(x1 + w/2)
            y = int(y1 + h/2)

            print("  ")
            print("GIMME GIMME GIMME AN INTEGER: ", x, y)
            print("  ")

            save_to_session('last_det', name)
            save_to_session('last_x', x)
            save_to_session('last_y', y)

            object_label = session.get('user_label', None)
            language = session.get('language', None)

            responseMessage = {"custom_detection_name":name, "custom_detection_confidence":confidence, "x":x, "y":y,"x_old":session.get('og_x'), "y_old":session.get('og_y'), "userInput":object_label, "language":language, "firstDet":session.get('og_det'), "rectangle":[y1,x1,y2,x2]} 
            responseType = "detection"
            

    elif requestType == "calculate":
        calc_type = requestMsg["type"]
        fileName = requestMsg["fileName"]
        imagePath = create_image_path(fileName)
        session[fileName] = imagePath
        print(calc_type)
        print(fileName)
        print(imagePath)

        if calc_type == "get_more_images":
            print("CALCULATING DIVERSIFIED IMAGES")
            images = diversify_images(imagePath)

            responseType = "diversified_images"
            responseMessage = images
        
        elif calc_type == "positions":
            print("CALCULATING POSITIONS")
            images = diversify_images_position(imagePath)

            responseType = "diversified_images_positions"
            responseMessage = images

        elif calc_type == "rotations":
            print("CALCULATING ROTATIONS")
            images = diversify_images_rotation(imagePath)

            responseType = "diversified_images_rotation"
            responseMessage = images

        elif calc_type == "positions_and_rotations":
            print("CALCULATING POSITIONS AND ROTATIONS")

            images = diversify_images_rotation_and_position(imagePath)

            responseType = "diversified_images_positions_and_rotations"
            responseMessage = images
        
        elif calc_type == "zoom":
            print("CALCULATING POSITIONS")
            images = diversify_images_zoom(imagePath)

            responseType = "diversified_images_zoom"
            responseMessage = images
    
    elif requestType == "save":
        
        # TODO
        img_url = requestMsg["image"]
        image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
        image_np = image64_to_numpy(image) # get numpy array representation of image
        fileName = requestMsg["fileName"]
        print("THE FILENAME: ", fileName)
        imagePath = save_image(image_np, fileName)
        
        session[fileName] = imagePath

        session['savedImage'] = 'True'

        save_to_session('og_x', session.get('last_x'))
        save_to_session('og_y', session.get('last_y'))
        save_to_session('og_det', session.get('last_det'))

        print(session.get('og_x'),session.get('og_y'),session.get('og_det'))

        responseMessage = {"x_old":"image saved"} 
        responseType = "success"

        print(" ")   
        print("DONE - DONE - DONE")  
        print(" ")  

    elif requestType == "sven_model_test":

        print("  ")
        print("Sven model fetch")
        print("  ")
        
        if requestMsg != "":

            img_url = requestMsg["image"]
            image = base64.b64decode(img_url.replace('data:image/png;base64,', ''))
            image_np = image64_to_numpy(image) # get numpy array representation of image 
            save_image(image_np, "temp_img.jpg")

            detection = sven_model_detect()
            print("  ")
            print("  ")
            print("Sven model detection:")
            print("  ")
            print(detection)
            print("  ")
            print("  ")

            name = detection[0]
            confidence = detection[1]
            y1 = detection[2]
            x1 = detection[3]
            y2 = detection[4]
            x2 = detection[5]

            w = x2 - x1
            h = y2 - y1

            x = int(x1 + w/2)
            y = int(y1 + h/2)

            save_to_session('last_det', name)
            save_to_session('last_x', x)
            save_to_session('last_y', y)

            object_label = session.get('user_label', None)
            language = session.get('language', None)

            responseMessage = {"custom_detection_name":name, "custom_detection_confidence":confidence, "x":x, "y":y,"x_old":session.get('og_x'), "y_old":session.get('og_y'), "userInput":object_label, "language":language, "firstDet":"testing", "rectangle":[y1,x1,y2,x2]} 
            responseType = "sven_model_test"

    elif requestType == "launch":
        testdetect()
        detectRoundTwo()
        responseType = "launch_ready"
        responseMessage = "launching app"


    msgRETURN = {"responseType":responseType, "responseMsg":responseMessage}

    return Response(json.dumps(msgRETURN))

#_______________________________________________________________
# OTHER METHODS
#
def create_image_path(fileName):
    path = os.path.join(os.getcwd(), "processed")
    imagePath = os.path.join(path, fileName)
    return imagePath
#_______________________________________________________________
# SAVE IMAGE
#
def save_image(data, fileName):
    print("SAVE IMAGE")
                                  # CREATE PATHS FOR IMAGE
    path = os.path.join(os.getcwd(), "processed")
    session["files_path"] = path
    print("PATH TO IMAGE: " + path)
    imagePath = os.path.join(path, fileName)

    save_to_session("image_path", imagePath)                            # SAVE PATH TO SESSION
  
    success = cv2.imwrite(imagePath, data)                              # SAVE IMAGE TO PATH
    print("SAVED IMAGE AS: " + fileName + " STATUS: " + str(success))
    return imagePath

def detect_file(imagePath):
    
    img_np = cv2.imread(imagePath)

    detection = detect_obj(img_np)
    return detection

def detect_old(path):
    img_np = cv2.imread(path)

    det = detect_obj(img_np)
    label = det[0]
    x = det[2]
    y = det[3]
    return [x,y,label]


def image64_to_numpy(data):
        fileName = "temp_img.jpg"

        path = os.path.join(os.getcwd(), "processed")
        imagePath = os.path.join(path, fileName)
        session['image_01_path'] = imagePath

        # read image (is read as numpy)
        with open(imagePath, 'wb') as f:
            f.write(data)   
        img_np = cv2.imread(imagePath)

        session['temp_image_path'] = imagePath

        return img_np

def image_np_to_64(image):
    path = session.get(image, None)

    with open(path, "rb") as image_file:
        image_as_64 = base64.b64encode(image_file.read())
    
    return image_as_64

#_______________________________________________________________
# DETECT FROM IMAGE AND SAVE POSITION AND DETECTION FOR LATER
#
def detect_and_save_from_image(image):

    print("DETECT AND SAVE")

    detection = detect_obj(image)

    print("DETECTION: " + str(detection))


    if detection != None:
        label = detection[0]
        confidence = detection[1]
        x = detection[2]
        y = detection[3]

        print("XPOS: " + str(x))
        print("YPOS: " + str(y))

        save_to_session("detection", label)
        save_to_session("confidence", confidence)
        save_to_session("x_pos", x)
        save_to_session("y_pos", y)

        print(label, confidence, x, y)


def generate_interaction():
    x = random.randint(1, 10)
    interaction = x % 2
    print("INTERACTION NUMBER: " + str(interaction))
    return interaction


#_______________________________________________________________
# 01 - 07 - QUESTIONNAIRE       -> save answers to questions
#

@app.route('/questionnaire1', methods=['POST'])
def questionnaire1():
    
    answer = request.form['answer']
    print("USER ANSWERED: " + answer) # TODO: SAVE ANSWER


    if "6" in answer:
        save_to_session('q6', answer)
        interaction_end()
        interaction_duration()

        file = session.get('csv', None)
        row = get_session_data()
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        page_id = "End"
    
    elif "5" in answer:
        save_to_session('q5', answer)

        page_id = "Q6"
    
    elif "4" in answer:
        save_to_session('q4', answer)

        page_id = "Q5"
    
    elif "3" in answer:
        save_to_session('q3', answer)

        page_id = "Q4"

    elif "2" in answer:
        save_to_session('q2', answer)

        page_id = "Q3"

    elif "1" in answer:
        save_to_session('q1', answer)
        page_id = "Q2"

    else:
        save_to_session('consent', answer)
        page_id = "Q1"


    return redirect(url_for('reroute', id=page_id))


def interaction_start():

    # Current GMT time in a tuple format
    current_GMT = time.gmtime()

    # ts stores timestamp
    ts = calendar.timegm(current_GMT)
    print("Current timestamp:", ts)

    start_time = datetime.fromtimestamp(ts, tz=None)

    print("Current time:", start_time)

    save_to_session('start_time', start_time)


def interaction_end():
    save_to_session('interaction_completed', 'True')
    # Current GMT time in a tuple format
    current_GMT = time.gmtime()

    # ts stores timestamp
    ts = calendar.timegm(current_GMT)
    print("Current timestamp:", ts)
    end_time = datetime.fromtimestamp(ts, tz=None)

    print("Current time:", end_time)

    save_to_session('end_time', end_time)

def interaction_duration():
    print('ENTERED INTERACTION DURATION')

    start = session.get('start_time', None)
    print('1', start)
    end = session.get('end_time', None)
    print('2', end)
    # Get the interval in minutes
    start = start.replace(tzinfo=None)
    end = end.replace(tzinfo=None)
    diff = end-start
    print(diff)
    diff_in_minutes = diff.total_seconds() / 60
    print('Difference between two datetimes in minutes:')
    print(diff_in_minutes)
    
    save_to_session('duration', diff_in_minutes)

def get_session_data():
    user_id = session.get('userID', None)
    true_detection = session.get('og_det', None)
    user_input = session.get('user_label', None)
    consentanswer = session.get('consent', None)
    q1_answer = session.get('q1', None)
    q2_answer = session.get('q2', None)
    q3_answer = session.get('q3', None)
    q4_answer = session.get('q4', None)
    q5_answer = session.get('q5', None)
    q6_answer = session.get('q6', None)
    interaction_type = session.get('interaction_type', None)
    interaction_duration = session.get('duration', None)
    interaction_completed = session.get('interaction_completed', None)

    return [user_id, true_detection, user_input, consentanswer, q1_answer, q2_answer,q3_answer, q4_answer, q5_answer, q6_answer, interaction_type, interaction_duration, interaction_completed]


#_______________________________________________________________
# PATH TO CSV FILE
#
header = ['userID', 'true detection', 'user input', 'consent', 'answer Q1', 'answer Q2', 'answer Q3', 'answer Q4', 'answer Q5', 'answer Q6', 'interaction type', 'interaction duration (min)', 'interaction completed']

with open(csvFile, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(header)



def generate_user_id():
    x = random.randint(0, 10000)
    y = random.choice(string.ascii_letters)
    userID = str(x) + str(y) + str(x)
    print("USER ID: " + userID)

    save_to_session('userID', userID)

    return userID



#_______________________________________________________________
# SAVE TO SESSION
#
def save_to_session(name, value):
    session[name] = value




def get_from_session(name):
    value = session.get(name, None)
    return value


if __name__ == "__main__":
    app.secret_key = 'the random string'
    app.run(host='0.0.0.0', port=80)