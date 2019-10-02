from styx_msgs.msg import TrafficLight
import rospy
import tensorflow as tf
import numpy as np
import time

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.categories = {1: "Red", 2: "Yellow", 3: "Green"}
        self.current_traffic_light = TrafficLight.UNKNOWN

        #MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
        MODEL_NAME = 'ssd_mobilenet_2018_01_28'
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
            # Get reference to those saved operations and placeholder variables via 
            # graph.get_tensor_by_name() method
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        self.current_traffic_light = TrafficLight.UNKNOWN

        with self.detection_graph.as_default():
            img_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, 
                                                           self.detection_classes, self.num_detections],
                                                           feed_dict={self.image_tensor: img_expanded})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            area = 0
            index = 0
            score_threshold = 0.5
            for i in range(len(boxes)):
                if scores[i] > score_threshold:
                    # for getting the closest traffic light 
                    temp = self.calculate_area(boxes[i])
                    if temp > area:
                        area = temp
                        index = i

            class_name = self.categories[classes[index]]
            if class_name == "Red":
                self.current_traffic_light = TrafficLight.RED
            elif class_name == "Green":
                self.current_traffic_light = TrafficLight.GREEN
            elif class_name == "Yellow":
                self.current_traffic_light = TrafficLight.YELLOW

            #rospy.loginfo('[Info] Traffic light state: {}'.format(class_name))
            #rospy.loginfo('[Info] state score: {}'.format(scores[index]))

        return self.current_traffic_light

    def calculate_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
