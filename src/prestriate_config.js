export const computeModel = 'prestriate'; //string; values: prestriate, striate

export const framesPerMinute = 60; //integer; number of video frames to capture per minute of video duration

export const objectCategories = ['person', 'vehicle', 'animal', 'object']; //array of strings; values: person, vehicle, animal, object

export const objectColours = ['blue', 'green', 'red', 'purple']; //array of strings; values: html colours; corresponds to objectCategories index

export const confidenceThreshold = 0.25; //integer between 0 and 1; only bounding boxes above this threshold will be drawn

export const boxLines = 0.0025; //integer between 0 and 1; multiplied by canvas width to determine thickness of bounding boxes