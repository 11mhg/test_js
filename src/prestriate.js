"use strict";

import * as tf from '@tensorflow/tfjs';
import config from './prestriate_config.js';

async function postprocess(
  outputs,
  anchors,
  numClasses,
  classNames,
  imageShape,
  maxBoxes,
  scoreThreshold,
  iouThreshold
) {

  const [boxes, boxScores] = yoloEval(outputs, anchors, numClasses, imageShape);

  let boxes_ = [];
  let scores_ = [];
  let classes_ = [];

  const _classes = tf.argMax(boxScores, -1);
  const _boxScores = tf.max(boxScores, -1);

  const nmsIndex = await tf.image.nonMaxSuppressionAsync(
    boxes,
    _boxScores,
    maxBoxes,
    iouThreshold,
    scoreThreshold
  );

  if (nmsIndex.size) {
    tf.tidy(() => {
      const classBoxes = tf.gather(boxes, nmsIndex);
      const classBoxScores = tf.gather(_boxScores, nmsIndex);

      classBoxes.split(nmsIndex.size).map(thisBox => {
        boxes_.push(thisBox.dataSync());
      });
      classBoxScores.dataSync().map(score => {
        scores_.push(score);
      });

      classes_ = _classes.gather(nmsIndex).dataSync();
    });
  }
  _boxScores.dispose();
  _classes.dispose();
  nmsIndex.dispose();

  boxes.dispose();
  boxScores.dispose();

  return boxes_.map((thisBox, i) => {
    const top = Math.max(0, thisBox[0]);
    const left = Math.max(0, thisBox[1]);
    const bottom = Math.min(imageShape[0], thisBox[2]);
    const right = Math.min(imageShape[1], thisBox[3]);
    const height = bottom - top;
    const width = right - left;
    return {
      box: [left, top, width, height],
      score: scores_[i],
      category: classNames[classes_[i]][0],
      object: classNames[classes_[i]][1]
    }
  });
}

function yoloEval(
  outputs,
  anchors,
  numClasses,
  imageShape
) {
  return tf.tidy(() => {
    let numLayers = 1;
    let inputShape;
    let anchorMask;

    numLayers = outputs.length;
    anchorMask = config.architecture_masks[numLayers];
    inputShape = outputs[0].shape.slice(1, 3).map(num => num * 32);

    const anchorsTensor = tf.tensor1d(anchors).reshape([-1, 2]);
    let boxes = [];
    let boxScores = [];

    for (let i = 0; i < numLayers; i++) {
      const [_boxes, _boxScores] = yoloBoxesAndScores(
        outputs[i],
        anchorsTensor.gather(tf.tensor1d(anchorMask[i], 'int32')),
        numClasses,
        inputShape,
        imageShape
      );

      boxes.push(_boxes);
      boxScores.push(_boxScores);
    };

    boxes = tf.concat(boxes);
    boxScores = tf.concat(boxScores);

    return [boxes, boxScores];
  });
}

function yoloBoxesAndScores(
  feats,
  anchors,
  numClasses,
  inputShape,
  imageShape
) {

  const [boxXy, boxWh, boxConfidence, boxClassProbs] = yoloHead(feats, anchors, numClasses, inputShape);

  let boxes = yoloCorrectBoxes(boxXy, boxWh, imageShape);
  boxes = boxes.reshape([-1, 4]);
  let boxScores = tf.mul(boxConfidence, boxClassProbs);
  boxScores = tf.reshape(boxScores, [-1, numClasses]);

  return [boxes, boxScores];
}

function yoloHead(
  feats,
  anchors,
  numClasses,
  inputShape
) {

  const numAnchors = anchors.shape[0];
  // Reshape to height, width, num_anchors, box_params.
  const anchorsTensor = tf.reshape(anchors, [1, 1, numAnchors, 2]);

  const gridShape = feats.shape.slice(1, 3); // height, width

  const gridY = tf.tile(tf.reshape(tf.range(0, gridShape[0]), [-1, 1, 1, 1]), [1, gridShape[1], 1, 1]);
  const gridX = tf.tile(tf.reshape(tf.range(0, gridShape[1]), [1, -1, 1, 1]), [gridShape[0], 1, 1, 1]);
  const grid = tf.concat([gridX, gridY], 3).cast(feats.dtype);

  feats = feats.reshape([gridShape[0], gridShape[1], numAnchors, numClasses + 5]);

  const [xy, wh, con, probs] = tf.split(feats, [2, 2, 1, numClasses], 3);
  // Adjust preditions to each spatial grid point and anchor size.
  const boxXy = tf.div(tf.add(tf.sigmoid(xy), grid), gridShape.reverse());
  const boxWh = tf.div(tf.mul(tf.exp(wh), anchorsTensor), inputShape.reverse());
  const boxConfidence = tf.sigmoid(con);

  let boxClassProbs;
  boxClassProbs = tf.sigmoid(probs);

  return [boxXy, boxWh, boxConfidence, boxClassProbs];
}

function yoloCorrectBoxes(
  boxXy,
  boxWh,
  imageShape
) {

  let boxYx = tf.concat(tf.split(boxXy, 2, 3).reverse(), 3);
  let boxHw = tf.concat(tf.split(boxWh, 2, 3).reverse(), 3);

  // Scale boxes back to original image shape.
  const boxMins = tf.mul(tf.sub(boxYx, tf.div(boxHw, 2)), imageShape);
  const boxMaxes = tf.mul(tf.add(boxYx, tf.div(boxHw, 2)), imageShape);

  const boxes = tf.concat([
    ...tf.split(boxMins, 2, 3),
    ...tf.split(boxMaxes, 2, 3)
  ], 3);

  return boxes;
}

async function _detectFromImage(
  model,
  image,
  maxBoxes,
  scoreThreshold,
  iouThreshold,
  numClasses,
  anchors,
  classNames,
  inputSize,
) {

  let outputs = tf.tidy(() => {
    const canvas = document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, inputSize, inputSize);

    let imageTensor = tf.browser.fromPixels(canvas, 3);
    imageTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));

    const predictions = model.predict(imageTensor);
    return predictions;
  });

  const boxes = await postprocess(
    outputs,
    anchors,
    numClasses,
    classNames,
//    image.constructor.name === 'HTMLVideoElement' ?
//      [image.videoHeight, image.videoWidth] :
//      [image.height, image.width],
    [inputSize, inputSize], //encompasses both video and image sizing via predefined input dimensions
    maxBoxes,
    scoreThreshold,
    iouThreshold
  );

  tf.dispose(outputs);
  return boxes;
}

async function _detectFromCanvas(
  model,
  imageCanvas,
  maxBoxes,
  scoreThreshold,
  iouThreshold,
  numClasses,
  anchors,
  classNames,
  inputSize,
) {

  let outputs = tf.tidy(() => {

    let imageTensor = tf.browser.fromPixels(imageCanvas, 3);
    imageTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));

    const predictions = model.predict(imageTensor);
    return predictions;
  });

  const boxes = await postprocess(
    outputs,
    anchors,
    numClasses,
    classNames,
    [inputSize, inputSize],
    maxBoxes,
    scoreThreshold,
    iouThreshold
  );

  tf.dispose(outputs);
  return boxes;
}

async function _detectFromTensor(
  imageTensor,
  model,
  maxBoxes = config.maxBoxes,
  scoreThreshold = config.scoreThreshold,
  iouThreshold = config.iouThreshold,
  numClasses = config.class_names.length,
  anchors = config.architecture_anchors,
  classNames = config.class_names,
  inputSize = config.input_size,
) {

  let outputs = tf.tidy(() => {
    const predictions = model.predict(imageTensor);
    return predictions;
  });

  const boxes = await postprocess(
    outputs,
    anchors,
    numClasses,
    classNames,
    [inputSize, inputSize],
    maxBoxes,
    scoreThreshold,
    iouThreshold
  );

  tf.dispose(outputs);
  return boxes;
}

async function _detectFromURL(
  model,
  imageURL,
  maxBoxes,
  scoreThreshold,
  iouThreshold,
  numClasses,
  anchors,
  classNames,
  inputSize,
) {

  function loadImage(url){
    return new Promise((resolve, reject) => {
      let image = document.createElement('img');
      image.crossOrigin = 'anonymous';
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = url;
    })
  }

  async function getImageData(url) {
    const image = await loadImage(url);

    const canvas = await document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = await canvas.getContext('2d');
    await ctx.drawImage(image, 0, 0, inputSize, inputSize);

    let output = tf.tidy(() => {
      let imageTensor = tf.browser.fromPixels(canvas, 3);
      imageTensor = imageTensor.expandDims(0).toFloat().div(tf.scalar(255));

      const predictions = model.predict(imageTensor);
      return predictions;
    });

    return output;
  }

  let outputs = await getImageData(imageURL);

  const boxes = await postprocess(
    outputs,
    anchors,
    numClasses,
    classNames,
    [inputSize, inputSize],
    maxBoxes,
    scoreThreshold,
    iouThreshold
  );

  tf.dispose(outputs);
  return boxes;
}

async function load(loadPath) {
  let model = await tf.loadLayersModel(loadPath);
  return {
    model,
    _detectFromTensor
  }
}

async function shards() {
  return config.architecture_shards;
}

const architecture = {
  load,
  shards
};

export default architecture;
