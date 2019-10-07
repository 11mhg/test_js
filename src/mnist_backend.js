//"use strict";

import * as tf from '@tensorflow/tfjs';
import tfserialize from './tfserialize';

// async function serializeModel(layersModel){

// }

// async function unserializeModel(ser_model){

// }

//coordinates the detections on the provided batch of images
async function computeBatch(
    myImages, myShapes, testSerialize = true, layersModel = false
) {
    //var myCompute = 'browser';
    var myCompute = 'scheduler';

    let myModel = null;
    if (layersModel) {
        myModel = await tf.loadLayersModel("http://127.0.0.1:5500/layers-js/model.json");
    } else {
        myModel = await tf.loadGraphModel("http://127.0.0.1:5500/graph-js/model.json");
    }

    let myResults = [];

    if (myCompute == 'local' || myCompute == 'scheduler') {

        //DCP computation prompts user for a wallet, then passes on the images
        await dcp.protocol.keychain.getKeystore();
        myResults = await _callDCP(myImages,myModel,layersModel,myCompute);

    } else if (myCompute == 'browser') {

        if (testSerialize && layersModel){
            console.log("Layers Model is being Serialized");
            let ser_model = await tfserialize.serialize(myModel,false);
            console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
            myModel = await tfserialize.deserialize(ser_model);
            console.log("Model has been deserialized and replaced");
        }else if (testSerialize && !layersModel){
            console.log("Graph Model is being Serialized");
            let ser_model = await tfserialize.serializeGraph(myModel);
            console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
            myModel = await tfserialize.deserializeGraph(ser_model);
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

async function batchify(allImages,batchSize){
    let batchified = []
    let curBatch = []
    for (let i=0;i<allImages.length;i++){
        curBatch.push(allImages[i]);
        if (curBatch.length >= batchSize){
            batchified.push(curBatch);
            curBatch = [];
        }
    }
    if (curBatch.length >=1){
        batchified.push(curBatch);
    }
    return batchified;
}

//distributed computation of frame batches (requires an available DCP network)
async function _callDCP(
    myImages,
    myModel,
    layersModel,
    myCompute,
    batchSize = 4,
) {

    let ser_model = null;

    if (layersModel){
        console.log("Layers Model is being Serialized");
        ser_model = await tfserialize.serialize(myModel,false);
        console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
    }else if (!layersModel){
        console.log("Graph Model is being Serialized");
        ser_model = await tfserialize.serializeGraph(myModel);
        console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
    }
    myModel = null;


    let myResults = [];
    
    myImages.push(myImages[0]);

    myImages = await batchify(myImages,batchSize);

    for (let i = 0; i < myImages.length; i++) {
        myImages[i] = JSON.stringify(myImages[i]);
    }

    let inferenceObj = {serialModel: ser_model, layersModel: layersModel};

    

    const workFunction = `async function (images, inferenceObj) {
    //load necessary DCP modules
    tf = require('tfjs');


    progress(1);


    return inferenceObj;
/*


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
    //define DCP generator, which receives array of image strings and a work function
    console.log(inferenceObj);

    const gen = dcp.compute.for(myImages, workFunction,[inferenceObj]);

    //define necessary modules, which must be available on the DCP module server 
    gen.requires('tensorflowdcp/tfjs');

    // gen.on('accepted', (thisOutput) => {
    //     console.log('Job accepted!');
    //     console.log(gen.id);
    // });

    // gen.on('complete', (thisOutput) => {
    //     console.log('Job complete!');
    // });

    //receives and collects the results of each worker thread
    gen.on('result', (thisOutput) => {
        myResults.push(thisOutput.result);
    });

    //worker thread title on DCP
    gen._generator.public = {
        name: 'Test'
    };

    if (myCompute == 'scheduler') {
        //sets gpu requirement and DCC payment, calls DCP generator on scheduler server
        //gen._generator.capabilities = {gpu: true};
        await gen.exec(0.0001);
        console.log('inference Object: ',myResults);
        console.log('are the models the same?: ',inferenceObj.serialModel === myResults[0].serialModel);
        throw "Here";
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