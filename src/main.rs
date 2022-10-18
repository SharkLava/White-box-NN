use ndarray::{Array, Array2, Axis, Slice};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::f64::consts;

use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;

/*
 fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
*/

/*
fn l1_norm(x: ArrayView2<f64>) -> f64 {
    x.fold(0., |acc, elem| acc + elem.abs())
}

fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.dot(&x).abs()
}
*/

fn relu_dev(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let result = x.map(|&i| if i > 0.0 { 1.0 } else { 0.0 });
    return result;
}

fn relu(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|elem| elem.max(0.00))
}

fn softmax(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    //let axis_sum = x.fold_axis(0., Axis(1),|sum,elem| sum +  consts::E.powf(*elem));
    let max_v = &x.iter().fold(0.0 / 0.0, |m, v| v.max(m));
    let x_e = x.map(|elem| consts::E.powf(*elem - max_v));
    let axis_sum = x.map_axis(Axis(0), |view| {
        1.0 / view.map(|i| consts::E.powf(*i - max_v)).sum()
    });
    let test = Array2::from_diag(&axis_sum);
    let result = x_e.dot(&test);
    return result;
}

fn foward_prop(
    x: &ndarray::Array2<f64>,
    w1: &ndarray::Array2<f64>,
    b1: &ndarray::Array2<f64>,
    w2: &ndarray::Array2<f64>,
    b2: &ndarray::Array2<f64>,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
) {
    let z1 = w1.dot(x) + b1;
    let a1 = relu(&z1);
    let z2 = w2.dot(&a1) + b2;
    let a2 = softmax(&z2);
    return (z1, a1, z2, a2);
}

fn back_prop(
    z1: &ndarray::Array2<f64>,
    a1: &ndarray::Array2<f64>,
    _z2: &ndarray::Array2<f64>,
    a2: &ndarray::Array2<f64>,
    w2: &ndarray::Array2<f64>,
    x: &ndarray::Array2<f64>,
    y: &ndarray::Array1<i32>,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
) {
    let m = y.shape()[0] as f64;
    let y_hot = one_hot(y);
    let d_z2 = a2 - y_hot;
    let d_w2 = 1.0 / m * (d_z2.dot(&a1.t()));
    //let db2 = 1.0 / m * dz2.fold_axis(Axis(1),0.0, |a , i| arr1(&[a]) + i);
    let db2 = Array::from_shape_vec(
        (10, 1),
        (d_z2.fold_axis(Axis(1), 0.0, |a, i| a + i)).to_vec(),
    )
    .unwrap();
    let d_b2 = 1.0 / m * db2;
    let d_z1 = w2.t().dot(&d_z2) * (relu_dev(z1));
    let d_w1 = 1.0 / m * d_z1.dot(&x.t());
    let d_b1 = Array::from_shape_vec(
        (10, 1),
        (d_z1.fold_axis(Axis(1), 0.0, |a, i| a + i)).to_vec(),
    )
    .unwrap();

    let _d_b1 = 1.0 / m * &d_b2;
    return (d_w1, d_b1, d_w2, d_b2);
}

fn update_params(
    w1: &ndarray::Array2<f64>,
    b1: &ndarray::Array2<f64>,
    w2: &ndarray::Array2<f64>,
    b2: &ndarray::Array2<f64>,
    dw1: &ndarray::Array2<f64>,
    db1: &ndarray::Array2<f64>,
    dw2: &ndarray::Array2<f64>,
    db2: &ndarray::Array2<f64>,
    alpha: f64,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
) {
    let w1_m = w1 - alpha * dw1;
    let b1_m = b1 - alpha * db1;
    let w2_m = w2 - alpha * dw2;
    let b2_m = b2 - alpha * db2;
    return (w1_m, b1_m, w2_m, b2_m);
}

fn predictions(x: ndarray::Array2<f64>) -> ndarray::Array1<i32> {
    let lenght = x.shape()[1];
    let mut results = Array::<i32, _>::zeros(lenght);
    for (i, col) in x.axis_iter(Axis(1)).enumerate() {
        let (max_idx, _max_val) =
            col.iter()
                .enumerate()
                .fold((0, col[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        results[i] = max_idx as i32;
    }
    return results;
}

fn gradient_descent(
    x: &ndarray::Array2<f64>,
    y: &ndarray::Array1<i32>,
    iter: i32,
    alpha: f64,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
) {
    let cols = x.shape()[1];
    println!("numbers of rows: {}", &cols);
    let z1 = Array2::<f64>::zeros((10, cols));
    let a1 = Array2::<f64>::zeros((10, cols));
    let z2 = Array2::<f64>::zeros((10, cols));
    let a2 = Array2::<f64>::zeros((10, cols));
    let mut foward_array = (z1, a1, z2, a2);
    let repeater = Array2::<f64>::ones((1, cols));
    let mut w1 = Array2::random((10, 784), Uniform::new(-0.5, 0.5));
    let mut b1 = (Array2::random((10, 1), Uniform::new(-0.5, 0.5))).dot(&repeater);
    let mut w2 = Array2::random((10, 10), Uniform::new(-0.5, 0.5));
    let mut b2 = (Array2::random((10, 1), Uniform::new(-0.5, 0.5))).dot(&repeater);

    for i in 0..iter {
        foward_array = foward_prop(&x, &w1, &b1, &w2, &b2);
        let (d_w1, d_b1, d_w2, d_b2) = back_prop(
            &foward_array.0,
            &foward_array.1,
            &foward_array.2,
            &foward_array.3,
            &w2,
            &x,
            &y,
        );
        let params = update_params(&w1, &b1, &w2, &b2, &d_w1, &d_b1, &d_w2, &d_b2, alpha);
        b1 = params.1;
        w1 = params.0;
        w2 = params.2;
        b2 = params.3;
        println!("Epoch: {}", i);
    }
    return (w1, b1, w2, b2);
}

fn one_hot(y: &ndarray::Array1<i32>) -> ndarray::Array2<f64> {
    // let len = x.shape()[0];
    let siz = y.iter().cloned().count();
    let max = itertools::max(y).unwrap();
    let max_usize: usize = *max as usize + 1;
    let mut board = Array2::<f64>::zeros((max_usize, siz));
    let mut count: usize = 0;
    for elem in y.iter() {
        board[[*elem as usize, count]] = 1 as f64;
        count = count + 1;
    }
    return board;
}

fn accuracy(y_pred: &ndarray::Array1<i32>, y_true: &ndarray::Array1<i32>) -> f64 {
    let zero = y_pred - y_true;
    let len = zero.shape()[0] as f64;
    let n_zero = zero.fold(
        0.0,
        |sum, elem| {
            if elem == &0 {
                sum + 1.0
            } else {
                sum + 0.0
            }
        },
    );
    return n_zero / len;
}

fn read_array_data(
    addr: &str,
    has_header: bool,
    rows: usize,
) -> Result<Array2<f64>, Box<dyn Error>> {
    // println!("{}", &p);
    let file = File::open(addr)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(has_header)
        .from_reader(file);
    let array_read: Array2<f64> = reader.deserialize_array2((rows, 785))?;
    Ok(array_read)
}

fn run_network() {
    let train_columns = 10000;
    let test_columns = 200;

    // use the test data to train because the train data was too big
    let train = read_array_data("data/mnist_test.csv", true, train_columns).unwrap();
    // using the first 200 entries in the train dataset as test
    let test = read_array_data("data/test.csv", true, test_columns).unwrap();

    // segmenting train data for train data performance test
    let x = &train.t();
    let y = &x.slice_axis(Axis(0), Slice::from(0..1));
    let x = &x.slice_axis(Axis(0), Slice::from(1..785)) / 255.0;
    let y_train = Array::from_iter(y.iter().map(|&val| val as i32));

    let x_t = &test.t();
    let y_t = &x_t.slice_axis(Axis(0), Slice::from(0..1));
    let x_t = &x_t.slice_axis(Axis(0), Slice::from(1..785)) / 255.0;
    let y_test = Array::from_iter(y_t.iter().map(|&val| val as i32));

    // println!("\n{}", &y_train);
    // println!("\n{}", &x);
    // println!("\n{}", &y_test);
    // println!("\n{}", &x);

    let (w1, b1, w2, b2) = gradient_descent(&x, &y_train, 1000, 0.3); // gradient_descent(x,y,inter,alpha)

    let (_z1, _a1, _z2, a2) = foward_prop(&x, &w1, &b1, &w2, &b2);

    let y_pred = predictions(a2);

    let b1_test = b1
        .slice_axis(Axis(1), Slice::from(0..test_columns))
        .to_owned();
    let b2_test = b2
        .slice_axis(Axis(1), Slice::from(0..test_columns))
        .to_owned();

    let (_z17, _a1t, _z2t, a2t) = foward_prop(&x_t, &w1, &b1_test, &w2, &b2_test);

    let y_pred_test = predictions(a2t);

    let acc = accuracy(&y_pred, &y_train);
    let acc_test = accuracy(&y_pred_test, &y_test);
    println!("accuracy  on trained dataset :  {}", &acc);
    println!("accuracy  on test dataset :  {}", &acc_test);

    // let soft_test = arr2(&[[100.0,2.9,4.3], [200.0,3.9,4.3], [1.0,300.9,9.3],[1.0,3.9,400.3],[1.0,7.9,4.3]]);
    // println!("{}", &soft_test);

    // let ss = softmax(&soft_test);
    // println!("{}", &ss);
}

fn main() {
    run_network();
}
