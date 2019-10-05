// prestriate demo application
// aisight inc
// this build is not for commercial use

//"use strict";
import * as tf from '@tensorflow/tfjs'
import * as backend from './backend.js'

let globalImageArray = null;
let globalImageShape = null;

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
    let currentImage = e.target;
    var canvas = document.getElementById('original_canvas');
    let new_sizes = aspectRatioResize(currentImage.naturalWidth, currentImage.naturalHeight, 200, 200);
    canvas.width = new_sizes[0];
    canvas.height = new_sizes[1];
    var ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, new_sizes[0], new_sizes[1]);

    let pixelData = ctx.getImageData(0, 0, currentImage.naturalWidth, currentImage.naturalHeight);
    let pixelTensor = await tf.browser.fromPixels(pixelData);

    globalImageArray = new Float32Array((await pixelTensor.data())).map(x => x / 255.);
    globalImageShape = pixelTensor.shape;

    var canvas = document.getElementById('predicted_canvas');
    canvas.width = new_sizes[0];
    canvas.height = new_sizes[1];
    var ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, new_sizes[0], new_sizes[1]);
}

function resizepixelarray(pixeldata, pixelshape, resize_width, resize_height) {
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');

    ctx.width = pixelshape[1];
    ctx.height = pixelshape[0];
    var imgdata = ctx.getImageData(0, 0, ctx.width, ctx.height);
    var px_data = new Uint8ClampedArray(pixeldata.map(x => x * 255.));

    imgdata.data.set(px_data);
    ctx.putImageData(imgdata, 0, 0);
    console.log(px_data);
    throw "stop";


}


function infer() {
    if (document.getElementById('uploaded_image').value.length < 4) {
        alert("Select photo for upload");
        return false;
    }

    var select = document.getElementById("model-type");
    var selectedVal = parseInt(select.options[select.selectedIndex].value);

    console.log(selectedVal);

    if (selectedVal == 1) {
        resizepixelarray(globalImageArray, globalImageShape, 480, 480);
        //backend.computeBatch()
    }

    return true;
}



document.getElementById("infer").onclick = infer;