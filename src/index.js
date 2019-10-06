// prestriate demo application
// aisight inc
// this build is not for commercial use

//"use strict";
import * as tf from '@tensorflow/tfjs'
import obj_backend from './obj_backend';
import mnist_backend from './mnist_backend';

let globalImageArray = null;
let globalImageShape = null;

console.log("Beginning Page");

document.querySelector('input[type="file"]').value = "";



window.addEventListener('load', function() {
    this.document.querySelector('input[type="file"]').addEventListener('change', function() {
        if (this.files && this.files[0]) {
            let image = new Image();
            image.src = URL.createObjectURL(this.files[0]);
            image.onload = imageIsLoaded;
        }
    })
})

function aspectRatioResize(width, height, maxWidth, maxHeight) {
    let ratio = 0;

    if (width > maxWidth) {
        ratio = maxWidth / width;
        height = height * ratio;
        width = width * ratio;
    }

    if (height > maxHeight) {
        ratio = maxHeight / height;
        height = height * ratio;
        width = width * ratio;
    }
    return [width, height];
}

async function imageIsLoaded(e) {


    //get which problem we are testing inference with
    var select = document.getElementById("model-type");
    var selectedVal = parseInt(select.options[select.selectedIndex].value);

    let output_sizes = [800, 600];
    if (selectedVal == 2 || selectedVal == 3) {
        output_sizes = [56, 56];
    }


    let currentImage = e.target;
    var canvas = document.getElementById('original_canvas');
    let new_sizes = aspectRatioResize(currentImage.naturalWidth, currentImage.naturalHeight, output_sizes[0], output_sizes[1]);
    canvas.width = currentImage.naturalWidth;
    canvas.height = currentImage.naturalHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, currentImage.naturalWidth, currentImage.naturalHeight);

    let pixelData = ctx.getImageData(0, 0, currentImage.naturalWidth, currentImage.naturalHeight);
    let pixelTensor = await tf.browser.fromPixels(pixelData);

    canvas.width = new_sizes[0];
    canvas.height = new_sizes[1];
    ctx.drawImage(currentImage, 0, 0, new_sizes[0], new_sizes[1]);

    globalImageArray = new Float32Array((await pixelTensor.data()));
    globalImageShape = pixelTensor.shape;

    var canvas = document.getElementById('predicted_canvas');
    canvas.width = new_sizes[0];
    canvas.height = new_sizes[1];
    var ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, new_sizes[0], new_sizes[1]);
}

function f32ToCanvas(pixeldata, pixelshape) {
    var canvas = document.createElement('canvas');
    canvas.width = pixelshape[1];
    canvas.height = pixelshape[0];
    var ctx = canvas.getContext('2d');

    var imageData = ctx.createImageData(pixelshape[1], pixelshape[0]);
    var data = imageData.data;
    var len = data.length;
    var i = 0;
    var t = 0;

    for (; i < len; i += 4) {
        data[i] = pixeldata[t];
        data[i + 1] = pixeldata[t + 1];
        data[i + 2] = pixeldata[t + 2];
        data[i + 3] = 255;
        t += 3;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}


//helper function which resizes pixels data to a particular width and height.
async function resizepixelarray(
    pixeldata, pixelshape, resize_width, resize_height) {
    var canvas = f32ToCanvas(pixeldata, pixelshape);

    var newCanvas = document.createElement('canvas');

    newCanvas.width = resize_width;
    newCanvas.height = resize_height;
    var newctx = newCanvas.getContext("2d");
    newctx.drawImage(canvas, 0, 0, newCanvas.width, newCanvas.height);

    let newData = newctx.getImageData(0, 0, resize_width, resize_height);
    let pixelTensor = await tf.browser.fromPixels(newData);


    let retData = new Float32Array((await pixelTensor.data()));
    let retShape = pixelTensor.shape;

    return [retData, retShape];
}

async function greyscalearray(
    pixeldata, pixelshape
) {
    let newpixeldata = new Float32Array(pixelshape[0] * pixelshape[1]);
    let newpixelshape = [pixelshape[0], pixelshape[1], 1];
    let ind = 0;
    for (; ind < newpixeldata.length; ind++) {
        let source_ind = ind * 3;
        newpixeldata[ind] = ((1 / 3) * pixeldata[source_ind]) + ((1 / 3) * pixeldata[source_ind + 1]) + ((1 / 3) * pixeldata[source_ind + 2]);
    }
    return [newpixeldata, newpixelshape]
}


async function infer() {

    //reset result text
    document.getElementById('result_text').innerHTML = '';



    if (document.getElementById('uploaded_image').value.length < 4) {
        alert("Select photo for upload");
        return false;
    }

    //get which problem we are testing inference with
    var select = document.getElementById("model-type");
    var selectedVal = parseInt(select.options[select.selectedIndex].value);

    console.log(selectedVal);

    //this one is object detection
    if (selectedVal == 1) {

        //test global image pixel
        // let predCanvas = document.getElementById('predicted_canvas');
        // predCanvas.width = globalImageShape[1];
        // predCanvas.height = globalImageShape[0];
        // let predctx = predCanvas.getContext("2d");
        // let fcanvas = f32ToCanvas(globalImageArray,globalImageShape);
        // predctx.drawImage(fcanvas,0,0);
        // return true;

        //resize to our ideal resize shape of 480,480,3
        let resize_ret = await resizepixelarray(globalImageArray, globalImageShape, 480, 480);

        //bump into array cause computebatch expects arrays
        let curImgArray = [resize_ret[0]];
        let curImgShape = [resize_ret[1]];

        // //TESTING THE PREDICTION CANVAS:
        // let predCanvas = document.getElementById('predicted_canvas');
        // predCanvas.width = curImgShape[0][1];
        // predCanvas.height = curImgShape[0][0];
        // let predctx = predCanvas.getContext("2d");
        // let fcanvas = f32ToCanvas(curImgArray[0],curImgShape[0]);
        // predctx.drawImage(fcanvas,0,0);

        // return true;

        //compute boxes with image array, with image shapes but do not test any serialization
        let resultBoxes = await obj_backend.computeBatch(curImgArray, curImgShape, false);


        for (let i = 0; i < resultBoxes.length; i++) {
            let bb = resultBoxes[i];
            await obj_backend.drawBoxestoContext(bb, document.getElementById('predicted_canvas').getContext("2d"));
        }

        //let user know on webpage that we are done.
        var text = document.createTextNode("Done Inferencing.");
        document.getElementById('result_text').appendChild(text);


    } else if (selectedVal == 2 || selectedVal == 3) {
        //This one is mnist testing
        //resize the image array to 28x28x3
        let resize_ret = await resizepixelarray(globalImageArray, globalImageShape, 28, 28);

        //greyscale to 28x28x1
        resize_ret = await greyscalearray(resize_ret[0], resize_ret[1]);
        let curImgArray = [resize_ret[0]];
        let curImgShape = [
            [784]
        ];

        //TESTING THE PREDICTION CANVAS:
        // let predCanvas = document.getElementById('predicted_canvas');
        // predCanvas.width = curImgShape[0][1];
        // predCanvas.height = curImgShape[0][0];
        // let predctx = predCanvas.getContext("2d");
        // let fcanvas = f32ToCanvas(curImgArray[0], curImgShape[0]);
        // predctx.drawImage(fcanvas, 0, 0);

        //compute mnist Digit with image array, with image shapes but do not test any serialization
        let doLayers = true;
        if (selectedVal == 3) {
            doLayers = false;
        }
        let resultDigits = await mnist_backend.computeBatch(curImgArray, curImgShape, false, doLayers);

        var text = document.createTextNode("Predicted digit is: " + resultDigits.toString());

        document.getElementById('result_text').appendChild(text);


        return true;
    }

    console.log("Done Inference");

    return true;
}



document.getElementById("infer").onclick = infer;