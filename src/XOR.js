const { createReadStream } = require('fs');
const csv = require('csv-parser');

const { matrix,
    exp,
    multiply,
    add,
    divide,
    det,
    transpose,
    subtract } = require('mathjs');

let x1 = [];
let x2= [];

let y= [];


//Simple perceptron
const sigmoid = (x) => {
    return divide(1, add(1,exp(multiply(-1,x))));
};
const sigmoidDerivative = (x) => {
    return multiply(sigmoid(x),subtract(1,sigmoid(x)));
};

let weights = matrix([[0, 0]]);
let bias = matrix([[1]]);

const forwordPropagation=(X,W,B) => {
    let Z1 = add(multiply(W,X),B);
    let Y = sigmoid(Z1);
    return {Y,Z1};
};

const backPropagation=(X,Y,Y_lable,Z1) => {
    let delta = subtract(Y,Y_lable);

    let dW = multiply(sigmoidDerivative(Z1),multiply(multiply(2,delta),transpose(X)));
    let dB = multiply(sigmoidDerivative(Z1),multiply(delta,matrix([[1]])));

    return {dW,dB};
};

const train = (X,Y_lable,W,B,learningRate,epoch) => {
    for(let i=0;i<epoch;i++){
        let {Y,Z1} = forwordPropagation(X,W,B);
        let {dW,dB} = backPropagation(X,Y,Y_lable,Z1,W,B);

        W = subtract(W,multiply(learningRate,dW));
        B = subtract(B,multiply(learningRate,dB));
    }
    return {W,B};
};


createReadStream('/home/banik/Desktop/Code/SimpleNN/DATA/XOR/Xor_Dataset.csv')
    .pipe(csv())
    .on('data', (data) => {
        x1.push(data.x);
        x2.push(data.y);
        y.push(data.z);
    })
    .on('end', () => {
        console.log('CSV file successfully processed');
        let training_x1 = x1.splice(0, x1.length * 0.7);
        let training_x2 = x2.splice(0, x2.length * 0.7);
        let training_y= y.splice(0, y.length * 0.7);

        let testing_x1 = x1.splice(0, x1.length);
        let testing_x2 = x2.splice(0, x2.length);
        let testing_y = y.splice(0, y.length);

        for (let i =0;i<training_x1.length;i++){
            let X = matrix([[training_x1[i]],[training_x2[i]]]);
            let Y_lable = matrix([[training_y[i]]]);
            let {W,B} = train(X,Y_lable,weights,bias,0.1,10);
            if(i%100===0){
                console.log(`Training -- ${i+1}`);
                let result=0;
                for(let k=0;k<testing_x1.length;k++){
                    let X_Test = matrix([[testing_x1[k]],[testing_x2[k]]]);
                    let Y_lable_Test = matrix([[testing_y[k]]]);
                    let { Y } = forwordPropagation(X_Test,W,B);
                    if(det(Y)>0.5){
                        if(det(Y_lable_Test)==1){
                            result++;
                        }
                    }else if(det(Y)<0.5){
                        if(det(Y_lable_Test)==0){
                            result++;
                        }
                    }
                }
                console.log(`Accuracy -- ${result/testing_x1.length}`);
            }
        }

    });

