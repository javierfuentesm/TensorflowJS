function getYs(xs, m, c) {
    return xs.mul(m).add(c)
}

const t1 = tf.tensor1d([1, 5, 10]);
const t2 = getYs(t1, 2, 1);
t2.print()
t2.dispose()


function normalize(tensor) {
    const max = tensor.max(); // 76
    const min = tensor.min(); // -5
    return tensor.sub(min).div(max.sub(min))

}

const t3 = tf.tensor1d([25, 76, 4, 23, -5, 22]);
normalize(t3).print()
t3.dispose()


for (let i = 0; i < 100; i++) {
    const lala = tf.tensor1d([1, 2, 3]);
    lala.dispose()
}


for (let i = 0; i < 100; i++) {
    tf.tidy(() => {
        tf.tensor1d([4, 5, 6]);
    });
}


console.log(tf.memory());
