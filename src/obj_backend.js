//"use strict";
//import * as config from './config';
import * as tf from '@tensorflow/tfjs';
import prestriate from './prestriate';
import config from './prestriate_config.js';
import tfserialize from './tfserialize';

/*
  await protocol.keychain.getKeystore();

  let myResult = 0;

  //define DCP generator, which receives array of image strings and a work function
  const gen = dcp.compute.for([newModel], async function (modelData) {
    progress(1);
    return modelData;
  });

  gen.on('accepted', (thisOutput) => {
    console.log('Job accepted!');
    console.log(gen.id);
  });

  gen.on('complete', (thisOutput) => {
    console.log('Job complete!');
  });

  //receives and collects the results of each worker thread
  gen.on('result', (thisOutput) => {

    myResult = thisOutput.result;

    console.log('-----------');
    console.log('-MODEL OUT-');
    console.log('-----------');
    console.log(myResult);
  });

  //worker thread title on DCP
  gen._generator.public = {
    name: 'aiSight Test'
  };

  await gen.exec(0.0001);
  return myResult;
*/
/*
  console.log('stringifying...');
  //let modelString = String(newModel);
  let modelString = newModel.toString();
  console.log('stringified!');

  console.log(modelString);

  //modelString = JSON.stringify(modelString);

  console.log('parsing...');
  let modelObject = JSON.parse(modelString);
  console.log('parsed!');

  console.log(modelObject);
*/
/*
  console.log('-------');
  console.log('-MODEL-');
  console.log('-------');
  console.log(newModel);

  let modelString = await objToString(newModel);

  console.log('--------');
  console.log('-STRING-');
  console.log('--------');
  console.log(modelString);

  let modelObject = await JSON.parse('{' + modelString + '}');

  console.log('--------');
  console.log('-OBJECT-');
  console.log('--------');
  console.log(modelObject);

  console.log('initiating...');
  let myModel = await prestriateTest.load(modelObject);
  console.log('initiated!');

  console.log(myModel);

  return myModel;
*/

//draw pixel array and object bounding box overlay onto canvas context
// function drawPixelArrayToContext(
//     myPixels,
//     myDimensions,
//     myContext,
//     myBoxes
// ) {

//     let savedMax = ((myPixels.length / 4) ** (1 / 2));

//     let originalWidth = myDimensions[0];
//     let originalHeight = myDimensions[1];

//     const maxWidth = myContext.canvas.width;
//     const maxHeight = myContext.canvas.height;

//     const savedRatio = Math.min(savedMax / originalWidth, savedMax / originalHeight);

//     const savedWidth = originalWidth * savedRatio;
//     const savedHeight = originalHeight * savedRatio;

//     const offsetWidth = (savedMax - savedWidth) / 2;
//     const offsetHeight = (savedMax - savedHeight) / 2;

//     const mysteryRatio = Math.min(maxWidth / savedWidth, maxHeight / savedHeight);

//     myContext.clearRect(0, 0, myContext.canvas.width, myContext.canvas.height);

//     let myImageData = new ImageData(myPixels, savedMax, savedMax);

//     let newContext = document.createElement('canvas').getContext('2d');
//     newContext.canvas.width = savedMax;
//     newContext.canvas.height = savedMax;

//     newContext.putImageData(myImageData, 0, 0);
//     newContext = drawBoxestoContext(myBoxes, newContext);
//     const targetWidth = savedWidth * mysteryRatio;
//     const targetHeight = savedHeight * mysteryRatio;

//     const targetOffsetWidth = (maxWidth - targetWidth) / 2;
//     const targetOffsetHeight = (maxHeight - targetHeight) / 2;

//     myContext.drawImage(newContext.canvas, offsetWidth, offsetHeight, savedWidth, savedHeight, targetOffsetWidth, targetOffsetHeight, targetWidth, targetHeight);

//     return myContext;
// }

//draw provided bounding boxes to a provided context
function drawBoxestoContext(
    myBoxes,
    myContext,
    myCategories = config.objectCategories,
    myThreshold = config.confidenceThreshold,
    myLines = config.boxLines,
    inputSize = config.input_size
) {

    let myWidthMod = myContext.canvas.width / inputSize;
    let myHeightMod = myContext.canvas.height / inputSize;

    let objectColours = ['blue', 'green', 'red', 'purple'];

    //set line pixel width using provided thickness multiplier
    myContext.lineWidth = myContext.canvas.width * myLines;
    for (let i = 0; i < myBoxes.length; i++) {
        let myIndex = myCategories.indexOf(myBoxes[i].category);

        //draw only boxes of the provided categories and confidence threshold
        if ((myIndex != -1) && (myBoxes[i].score >= myThreshold)) {
            myBoxes[i].box[0] = myBoxes[i].box[0] * myWidthMod;
            myBoxes[i].box[1] = myBoxes[i].box[1] * myHeightMod;
            myBoxes[i].box[2] = myBoxes[i].box[2] * myWidthMod;
            myBoxes[i].box[3] = myBoxes[i].box[3] * myHeightMod;

            //set draw colours from array in config file
            myContext.strokeStyle = objectColours[myIndex];
            myContext.fillStyle = objectColours[myIndex];

            //draw bounding box
            myContext.beginPath();
            myContext.rect(...myBoxes[i].box);
            myContext.stroke();

            //draw filled label square above bounding box
            myContext.beginPath();
            myContext.rect(myBoxes[i].box[0], myBoxes[i].box[1], myBoxes[i].box[2], myContext.lineWidth * 10);
            myContext.fill()

            //draw object class text to label
            myContext.font = (myContext.lineWidth * 7) + 'px Courier New';
            myContext.textBaseline = 'top';
            myContext.fillStyle = '#FFFFFF';
            myContext.fillText(myBoxes[i].object, myBoxes[i].box[0], myBoxes[i].box[1]);
        }
    }

    return myContext;
}

//coordinates the detections on the provided batch of images
async function computeBatch(
    myImages, myShapes, testSerialize = true
) {
    var myCompute = 'browser';

    let myResults = [];

    if (myCompute == 'local' || myCompute == 'scheduler') {

        //DCP computation prompts user for a wallet, then passes on the images
        await protocol.keychain.getKeystore();
        myResults = await _callDCP(myImages);

    } else if (myCompute == 'browser') {


        let myModel = await prestriate.load('https://aisight.ca/prestriate/model/model.json');
        if (testSerialize){
            console.log("Model is being Serialized");
            let ser_model = await tfserialize.serialize(myModel.model,false);
            let model = await tfserialize.deserialize(ser_model);
            myModel.model = model;
            console.log("Model has been deserialized and replaced");
        }


        for (let i = 0; i < myImages.length; i++) {

            //converts images from float32array to regular array
            let thisImage = myImages[i];
            let thisShape = myShapes[i];

            //converts image array to an appropriate tensor prior to loading into the model
            tf.setBackend('webgl');
            let imageTensor = await tf.tidy(() => {
                let myTensor = tf.tensor(thisImage, thisShape);
                return myTensor.expandDims(0).toFloat().div(255.);
            });

            //calls loaded object detection model with the provided image tensor and adds the results to an array
            let thisPrediction = await myModel._detectFromTensor(imageTensor,myModel.model);
            //      let thisPrediction = await global.model.detectFromTensor(imageTensor);

            //prediction tensor gives boxes between 0 and 480, need to normalize in future
            imageTensor.dispose();
            myResults.push(thisPrediction);
        }
    }

    return myResults;
}

//distributed computation of frame batches (requires an available DCP network)
async function _callDCP(
    myImages,
    myCompute,
    myArchitecture
) {

    let myResults = [];

    //convert all pixel arrays to strings before passing to DCP
    for (let i = 0; i < myImages.length; i++) {
        myImages[i] = myImages[i].toString();
    }

    let newModel = await prestriate.load();

    let modelData = JSON.stringify(newModel);

    console.log(modelData);

    /*
      //worker function to be passed to DCP, in the form of a template literal string
      const workFunction = `async function (imageData) {

        //load necessary DCP modules
        tf = require('tfjs');
        nn = require('${myArchitecture}.js');

        progress(0.2);

        //convert pixel string into array, remove alpha channel, set input size
        let thisImage = JSON.parse('[' + imageData + ']');
        thisImage = thisImage.filter((e, i) => (i + 1) % 4);
        const imageWidth = (thisImage.length / 3) ** (1/2);

        progress(0.4);

        //convert pixel array into properly configured tensor
        //tf.setBackend('webgl');
        let imageTensor = await tf.tidy(() => {
          let myTensor = tf.tensor(thisImage, [imageWidth, imageWidth, 3]);
          return myTensor.expandDims(0).toFloat().div(tf.scalar(255));
        });

        progress(0.6);

        //load architecture
        if (!global.hasOwnProperty('model')) {
            global.model = await nn.default.load();
        }

        progress(0.8);

        //pass pixel array tensor to model for detection 
        let thisPrediction = await model.detectFromTensor(imageTensor);
        imageTensor.dispose();

        progress(1);

        return thisPrediction;
      }`;
    */
    //WORKER FUNCTION TEST AREA
    const workFunction = `async function (imageData) {

    progress(1);

    return imageData;
/*
    //load necessary DCP modules
    tf = require('tfjs');
    nn = require('${myArchitecture}.js');

    setInterval(progress(), 5000);

    progress(0.2);

    //convert pixel string into array, remove alpha channel, set input size
    let thisImage = JSON.parse('[' + imageData + ']');
    thisImage = thisImage.filter((e, i) => (i + 1) % 4);
    const imageWidth = (thisImage.length / 3) ** (1/2);

    //convert modelData to model;

    progress(0.4);

    //convert pixel array into properly configured tensor
    //tf.setBackend('webgl');
    let imageTensor = await tf.tidy(() => {
      let myTensor = tf.tensor(thisImage, [imageWidth, imageWidth, 3]);
      return myTensor.expandDims(0).toFloat().div(tf.scalar(255));
    });

    progress(0.6);

    //load architecture
    let model = await nn.load('https://aisight.ca/prestriate/model/model.json');

//    if (!global.hasOwnProperty('model')) {
//      global.model = await nn.default.load();
//    }

    progress(0.8);

    //pass pixel array tensor to model for detection 
    //let thisPrediction = await model.detectFromTensor(imageTensor);
    //imageTensor.dispose();

    progress(1);

    //return thisPrediction;
    return imageTensor;
*/
  }`;
    console.log(workFunction);
    //define DCP generator, which receives array of image strings and a work function
    const gen = compute.for([myImages], workFunction);

    //define necessary modules, which must be available on the DCP module server 
    gen.requires('tensorflowdcp/tfjs');
    gen.requires('prestriatedcp/prestriate.js');

    gen.on('accepted', (thisOutput) => {
        console.log('Job accepted!');
        console.log(gen.id);
    });

    gen.on('complete', (thisOutput) => {
        console.log('Job complete!');
    });

    //receives and collects the results of each worker thread
    gen.on('result', (thisOutput) => {
        console.log('Job results:');
        console.log(thisOutput.result);
        myResults.push(thisOutput.result);
    });

    //worker thread title on DCP
    gen._generator.public = {
        name: 'aiSight Test'
    };

    if (myCompute == 'scheduler') {
        //sets gpu requirement and DCC payment, calls DCP generator on scheduler server
        //gen._generator.capabilities = {gpu: true};
        await gen.exec(0.0001);
        return myResults;
    } else if (myCompute == 'local') {
        //sets number of local cores, calls DCP generator on the local machine
        let workerNumber = config.computeWorkers;
        await gen.localExec(workerNumber);
        return myResults;
    }
}

//general-use function that loads, converts, and run detections on all provided media files
async function handleFiles(
    newFiles,
    myCompute = config.computePlatform,
    batchSize = config.batchSlices
) {

    let myFiles = [...newFiles];

    let myImages = [];

    //identifies and creates URLs for all provided media files, before passing them on for loading and conversion to pixel arrays
    for (let i = 0; i < myFiles.length; i++) {
        if (myFiles[i].type.indexOf('image/') !== -1) {
            let myUrl = await (URL || webkitURL).createObjectURL(myFiles[i]);
            let newImage = await loadImage(myUrl, true);
            (URL || webkitURL).revokeObjectURL(myUrl);
            myImages.push(newImage);
        } else if (myFiles[i].type.indexOf('video/') !== -1) {
            let myUrl = await (URL || webkitURL).createObjectURL(myFiles[i]);
            let newFrames = await loadVideo(myUrl, true);
            (URL || webkitURL).revokeObjectURL(myUrl);
            myImages = myImages.concat(newFrames);
        } else {
            //uploaded file is not a valid type
        }
    }

    let myBoxes = [];

    //passes pixel array frames on for detection in a series of batches 
    while (myImages.length > 0) {
        let batchBoxes = await computeBatch(myImages.splice(0, batchSize));
        myBoxes = myBoxes.concat(batchBoxes);
    }

    return myBoxes;
}

const platform = {
    //     drawPixelArrayToContext,
    drawBoxestoContext,
    //     drawImageToContext,
    //     processImage,
    //     processVideo,
    //     loadImage,
    //     loadVideo,
    computeBatch
    //     handleFiles,
    //     modelTest,
};

export default platform;