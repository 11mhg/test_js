"use strict";

import * as tf from '@tensorflow/tfjs';

// async function serializeModel(layersModel){

// }

// async function unserializeModel(ser_model){

// }

//coordinates the detections on the provided batch of images
async function computeBatch(
    myImages, myShapes, testSerialize = true, layersModel = false
) {
    var myCompute = 'browser';

    let myResults = [];

    if (myCompute == 'local' || myCompute == 'scheduler') {

        //DCP computation prompts user for a wallet, then passes on the images
        await protocol.keychain.getKeystore();
        myResults = await _callDCP(myImages);

    } else if (myCompute == 'browser') {
        let myModel = null;
        if (layersModel) {
            myModel = await tf.loadLayersModel("http://127.0.0.1:5500/layers-js/model.json");
        } else {
            myModel = await tf.loadGraphModel("http://127.0.0.1:5500/graph-js/model.json");
        }

        // if (testSerialize){
        //     let ser_model = await serializeModel(myModel);
        //     myModel = await unserializeModel(ser_model);
        // }


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
            let thisPrediction = await myModel.predict(imageTensor);
            let predictions = await tf.argMax(thisPrediction, -1).data();
            //      let thisPrediction = await global.model.detectFromTensor(imageTensor);

            //prediction tensor gives boxes between 0 and 480, need to normalize in future
            imageTensor.dispose();
            thisPrediction.dispose();
            myResults.push(predictions[0]);
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



const platform = {
    computeBatch
};

export default platform;