let net, webcam, str;
const imgEl = document.getElementById('img')
const descripcion = document.getElementById('descripcion_imagen')
const webcamElement = document.getElementById('webcam')
const consola = document.getElementById('consola')
const consola2 = document.getElementById('consola2')
const classifier = knnClassifier.create()


const app = async () => {
    net = await mobilenet.load()
    displayImagePrediction()
    webcam = await tf.data.webcam(webcamElement);

    while (true) {
        const img = await webcam.capture()
        const result = await net.classify(img)
        consola.innerHTML = `prediction ${result[0].className}  probability: ${result[0].probability}`
        const activation = net.infer(img, "conv_preds")
        try {
            const result2 = await classifier.predictClass(activation)
            const classes = ["Untrained", "Airpods", "Javier", "OK", "Rock"]
            consola2.innerHTML = `Prediccion personalizada ${classes[result2.label]} Probabilidad ${result2.confidences[result2.label]}`
        } catch (e) {
            consola2.innerText = "Untrained"
        }
        img.dispose()
        await tf.nextFrame()
    }

}

imgEl.onload = async () => {
    displayImagePrediction()
}

const addExample = async (classId) => {
    const img = await webcam.capture()
    const activation = net.infer(img, true)
    classifier.addExample(activation, classId)
    img.dispose()

}

const guardarModelo = () => {
    str = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    download(str, 'modelo.json', 'text/plain');
}

const cargarModelo = () => {
    const result= fetch("modelo.json")
        .then(response => response.json())
        .then(json => classifier.setClassifierDataset(Object.fromEntries(json.map(([label, data, shape]) => [label, tf.tensor(data, shape)]))));


}

const download = (content, fileName, contentType) => {
    var a = document.createElement("a");
    var file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}

const cambiarImagen = async () => {
    let count = 0;
    count++
    imgEl.src = 'https://picsum.photos/200/300?random=' + count
}
const displayImagePrediction = async () => {
    try {
        const result = await net.classify(imgEl)
        console.log(result)

        descripcion.innerHTML = JSON.stringify(result)
    } catch (error) {
        console.log(error)
    }
}
app()
