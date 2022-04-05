const { createReadStream } = require('fs');
const csv = require('csv-parser');

const { exp,add, matrix, divide, subtract,multiply, identity, ones, dot, transpose,dotMultiply } = require('mathjs');

let x1 = [];
let x2= [];

let y= [];

let weights1 =ones(2,2);
let bias1 = ones(2,1);

let weights2 = ones(1,2);
let bias2 = ones(1,1);


const sigmoid = (X) => {
    let sig =X.map(x => {
        let temp =1/(1+exp(-x));
        return temp;
    });


    return sig;
};

const relu = (X)=>{
    let relu = X.map(x => {
        if(x<0){
            return 0;
        }else{
            return x;
        }
    });
    return relu;
};


const derivative_relu = (X)=>{
    let dev_relu = X.map(x => {
        if(x<0){
            return 0;
        }else{
            return 1;
        }
    });
    return dev_relu;
};

const devrivative_sigmoid = (X) => {
    let sig = sigmoid(X);
    let dev_sig = sig.map(x => {
        let temp = x*(1-x);
        return temp;
    });

    return dev_sig;
};


const forwordPropagation=(X,weights1,bias1,weights2,bias2) => {
    let Z1 = add(multiply(weights1,X),bias1);
    let X2 = sigmoid(Z1);
    
    let Z2 = add(multiply(weights2,X2),bias2);
    let Y = relu(Z2);

    return { Z1,X2,Z2,Y};
};

let a = matrix([[1],[2]]);
let b = matrix([[3]]);
let c = matrix([[1]]);

const backPropagation=(Y,Y_lable,X,Z1,X2,Z2)=>{
    let E = subtract(Y,Y_lable);
    let dE_dY = multiply(2,E);
    E = E.map(x => {
        return x**2;
    });
    let dY_dZ2 = devrivative_sigmoid(Z2);
    let dZ2_dW2 = X2;

    let dE_dW2 = multiply(dY_dZ2 ,multiply(dE_dY,transpose(dZ2_dW2)));
    let dE_dB2 = multiply(dY_dZ2,dE_dY);

    let dX2_dZ1 = derivative_relu(Z1);
    let dZ1_dW1 = X;

    let dE_dW1 = multiply(dX2_dZ1,dotMultiply(dE_dW2,transpose(dZ1_dW1)));
    let dE_dB1 = multiply(dX2_dZ1,dE_dY);

    return { dE_dW1,dE_dB1,dE_dW2,dE_dB2};
};

const train = (X,Y_lable,learning_rate,epoch)=>{

    for(let i = 0 ; i<epoch;i++){
        let { Z1,X2,Z2,Y } = forwordPropagation(X,weights1,bias1,weights2,bias2);
        let { dE_dB1,dE_dB2,dE_dW1,dE_dW2 } = backPropagation(Y,Y_lable,X,Z1,X2,Z2);
        weights1 = add(weights1,multiply(learning_rate,dE_dW1));
        weights2 = add(weights2,multiply(learning_rate,dE_dW2));
        bias1 = add(bias1,multiply(learning_rate,dE_dB1));
        bias2 = add(bias2,multiply(learning_rate,dE_dB2));
    }

    return { weights1,bias1,weights2,bias2};
}


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
            let X = matrix([[parseInt(training_x1[i])],[parseInt(training_x2[i])]]);
            let Y_lable = matrix([[parseInt(training_y[i])]]);
            let RESULT = train(X,Y_lable,0.01,10);
            if(i%100 == 0){
                let result =0;
                for(let j=0;j<testing_x1.length;j++){
                    let X = matrix([[parseInt(testing_x1[j])],[parseInt(testing_x2[j])]]);
                    let Y_lable = matrix([[parseInt(testing_y[j])]]);
                    let FORWORD = forwordPropagation(X,weights1,bias1,weights2,bias2);
                    if(FORWORD.Y.get([0,0]) > 0.5 && Y_lable.get([0,0]) == 1){
                        result++;
                    }
                    if(FORWORD.Y.get([0,0]) < 0.5 && Y_lable.get([0,0]) == 0){
                        result++;
                    }
                }
                console.log(`Accuracy at ${i} --- ${(result/testing_x1.length)*100}%`);
                console.log(`Weights1 ---- ${RESULT.weights1}`);
                console.log(`Weights2 ---- ${RESULT.weights2}`);
                console.log(`Bias1 --- ${RESULT.bias1}`);
                console.log(`Bias2 ---  ${RESULT.bias2}`);
            }
        }
    });

