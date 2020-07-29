
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

const visualizarDatos = async (data) => {
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

const crearModelo = ()=>{
    const modelo = tf.sequential()
    modelo.add(tf.layers.dense({inputShape:[1],units: 1, useBias:true}))
    modelo.add(tf.layers.dense({units:1,useBias: true}))
    return modelo
}

const run = async () => {
    visualizarDatos(await getData())
    crearModelo()
}

run()
