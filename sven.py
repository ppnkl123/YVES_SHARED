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