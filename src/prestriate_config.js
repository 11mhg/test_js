const computeModel = 'prestriate'; //string; values: prestriate, striate

const framesPerMinute = 60; //integer; number of video frames to capture per minute of video duration

const objectCategories = ['person', 'vehicle', 'animal', 'object']; //array of strings; values: person, vehicle, animal, object

const objectColours = ['blue', 'green', 'red', 'purple']; //array of strings; values: html colours; corresponds to objectCategories index

const confidenceThreshold = 0.25; //integer between 0 and 1; only bounding boxes above this threshold will be drawn

const boxLines = 0.0025; //integer between 0 and 1; multiplied by canvas width to determine thickness of bounding boxes

const architecture_anchors = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319];

const architecture_masks = {
  "3": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
  "2": [[3, 4, 5], [1, 2, 3]]
};


// CONFIG INFERENCE SETTINGS

const max_boxes = 125;

const score_threshold = 0.25;

const iou_threshold = 0.5;

const input_size = 480;

// CONFIG DATASET CLASSES

const class_names = [
  ['person', 'person'],
  ['vehicle', 'bicycle'],
  ['vehicle', 'car'],
  ['vehicle', 'motorbike'],
  ['vehicle', 'aeroplane'],
  ['vehicle', 'bus'],
  ['vehicle', 'train'],
  ['vehicle', 'truck'],
  ['vehicle', 'boat'],
  ['object', 'traffic light'],
  ['object', 'fire hydrant'],
  ['object', 'stop sign'],
  ['object', 'parking meter'],
  ['object', 'bench'],
  ['animal', 'bird'],
  ['animal', 'cat'],
  ['animal', 'dog'],
  ['animal', 'horse'],
  ['animal', 'sheep'],
  ['animal', 'cow'],
  ['animal', 'elephant'],
  ['animal', 'bear'],
  ['animal', 'zebra'],
  ['animal', 'giraffe'],
  ['object', 'backpack'],
  ['object', 'umbrella'],
  ['object', 'handbag'],
  ['object', 'tie'],
  ['object', 'suitcase'],
  ['object', 'frisbee'],
  ['object', 'skis'],
  ['object', 'snowboard'],
  ['object', 'sports ball'],
  ['object', 'kite'],
  ['object', 'baseball bat'],
  ['object', 'baseball glove'],
  ['object', 'skateboard'],
  ['object', 'surfboard'],
  ['object', 'tennis racket'],
  ['object', 'bottle'],
  ['object', 'wine glass'],
  ['object', 'cup'],
  ['object', 'fork'],
  ['object', 'knife'],
  ['object', 'spoon'],
  ['object', 'bowl'],
  ['object', 'banana'],
  ['object', 'apple'],
  ['object', 'sandwich'],
  ['object', 'orange'],
  ['object', 'broccoli'],
  ['object', 'carrot'],
  ['object', 'hot dog'],
  ['object', 'pizza'],
  ['object', 'donut'],
  ['object', 'cake'],
  ['object', 'chair'],
  ['object', 'sofa'],
  ['object', 'pottedplant'],
  ['object', 'bed'],
  ['object', 'diningtable'],
  ['object', 'toilet'],
  ['object', 'tvmonitor'],
  ['object', 'laptop'],
  ['object', 'mouse'],
  ['object', 'remote'],
  ['object', 'keyboard'],
  ['object', 'cell phone'],
  ['object', 'microwave'],
  ['object', 'oven'],
  ['object', 'toaster'],
  ['object', 'sink'],
  ['object', 'refrigerator'],
  ['object', 'book'],
  ['object', 'clock'],
  ['object', 'vase'],
  ['object', 'scissors'],
  ['object', 'teddy bear'],
  ['object', 'hair drier'],
  ['object', 'toothbrush']
];

const config = {
    computeModel,
    framesPerMinute, //integer; number of video frames to capture per minute of video duration
    objectCategories, //array of strings; values: person, vehicle, animal, object
    objectColours, //array of strings; values: html colours; corresponds to objectCategories index
    confidenceThreshold, //integer between 0 and 1; only bounding boxes above this threshold will be drawn
    boxLines, //integer between 0 and 1; multiplied by canvas width to determine thickness of bounding boxes
    architecture_anchors,
    architecture_masks,
    class_names,
    max_boxes,
    score_threshold,
    iou_threshold,
    input_size
};

export default config;
