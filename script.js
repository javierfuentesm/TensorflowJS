let modelo, stopTraining

const getData = async () => {
    const datosCasasR = await fetch("data.json")
    const datosCasas = await datosCasasR.json()
    let datosLimpios = datosCasas.map(
        casa => ({precio: casa.Precio, cuartos: casa.NumeroDeCuartosPromedio})
    )
    datosLimpios = datosLimpios.filter(casa => (casa.precio != null && casa.cuartos != null))
    console.log(datosLimpios)
    return datosLimpios
}

const visualizarDatos = (data) => {
    const valores = data.map(d => ({x: d.cuartos, y: d.precio}))
    tfvis.render.scatterplot({
            name: 'Cuartos vs Precio'
        },
        {values: valores}, {
            xLabel: 'Cuartos',
            yLabel: 'Precio',
            height: 300
        }
    )
}

const crearModelo = () => {
    const modelo = tf.sequential()
    modelo.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))
    modelo.add(tf.layers.dense({units: 1, useBias: true}))
    return modelo
}
const convertirDatosaTensores =
    (data) => {
        return tf.tidy(() => {
            tf.util.shuffle(data)
            const entradas = data.map(d => d.cuartos)
            const etiquetas = data.map(d => d.precio)
            const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1])
            const tensorEtiquetas = tf.tensor2d(etiquetas, [etiquetas.length, 1])

            const entradasMax = tensorEntradas.max()
            const entradasMin = tensorEntradas.min()
            const etiquetasMax = tensorEtiquetas.max()
            const etiquetasMin = tensorEtiquetas.min()

            // (dato-min)/(max-min)

            const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin))
            const etiquetasNormalizadas = tensorEtiquetas.sub(entradasMin).div(etiquetasMax.sub(etiquetasMin))
            return {
                entradas: entradasNormalizadas,
                etiquetas: etiquetasNormalizadas,
                entradasMax,
                entradasMin,
                etiquetasMax,
                etiquetasMin
            }


        })
    }

const optimizador = tf.train.adam()
const funcion_perdida = tf.losses.meanSquaredError
const metricas = ['mse']

const entrenarModelo = async (model, inputs, labels) => {

    model.compile({
        optimizer: optimizador,
        loss: funcion_perdida,
        metrics: metricas
    })

    const surface = {name: 'show.history live', tab: 'Training'}
    const tamanioBatch = 28
    const epochs = 50
    const history = []

    return await model.fit(inputs, labels, {
        tamanioBatch,
        epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, log) => {
                history.push(log)
                tfvis.show.history(surface, history, ['loss', 'mse'])
                if (stopTraining) {
                    modelo.stopTraining = true
                }
            }
        }
    })

}

//Almacenar el modelo

const guardarModelo = async () => {
    const saveResult = await modelo.save('downloads://modelo-regresion')
}

const cargarModelo = async () => {
    const uploadJSONInput = document.getElementById('upload-json')
    const uploadWeightsInput = document.getElementById('upload-weights')

    modelo = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]))
    console.log('Modelo Cargado')

}

// mostrar curva de inferencia

const verCurvaInferencia = async () => {
    const data = await getData()
    const tensorData = await convertirDatosaTensores(data)
    const {entradasMax, entradasMin, etiquetasMin, etiquetasMax} = tensorData

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100)
        const preds = modelo.predict(xs.reshape([100, 1]))
        const desnormX = xs.mul(entradasMax.sub(entradasMin)).add(entradasMin)
        const desnormY = preds.mul(etiquetasMax.sub(etiquetasMin)).add(etiquetasMin)

        //Desnormalizar la informacion
        return [desnormX.dataSync(), desnormY.dataSync()]
    })

    const puntosPrediccion = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    })
    const puntosOriginales = data.map((d) => {
        return {x: d.cuartos, y: d.precio}
    })

    tfvis.render.scatterplot(
        {name: 'Predicciones vd Originales'},
        {values: [puntosOriginales, puntosPrediccion], series: ['originales', 'predicciones']},
        {
            xLabel: 'Cuartos',
            yLabel: 'Precio',
            height: 300
        }
    )
}


const run = async () => {
    const data = await getData()
    visualizarDatos(data)
    modelo = crearModelo()
    const tensorData = convertirDatosaTensores(data)
    const {entradas, etiquetas} = tensorData
    entrenarModelo(modelo, entradas, etiquetas)
}

run()
