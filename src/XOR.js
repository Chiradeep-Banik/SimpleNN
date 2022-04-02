const { createReadStream } = require('fs');
const csv = require('csv-parser');

const { matrix,
    exp,
    multiply,
    add,
    divide,
    det,
    subtract, 
    inv,
    map,
    transpose} = require('mathjs');

let x1 = [];
let x2= [];

let y= [];

let weights = matrix([[1],[1]]);
let bias = matrix([[1]]);

//Simple perceptron
const sigmoid = (x) => {
    return divide(1, add(1,exp(multiply(-1,x))));
};
const sigmoidDerivative = (x) => {
    let sig = sigmoid(x);
    return multiply(sig,subtract(1,sig));
};

const forwordPropagation=(X,weights,bias) => {
    let Z1 = add(multiply(X,weights),bias);
    let Y = sigmoid(Z1);
    return {Y,Z1};
};

const backPropagation=(Y,Y_lable,X)=>{
    let dY = subtract(Y,Y_lable);
    let dZ1 = sigmoidDerivative(Y);
    let denominator_W = multiply(transpose(X),dZ1);
    denominator_W = map(denominator_W,(x)=>{
        return 1/x;
    });
    let dW = multiply(denominator_W,dY);
    let dB = divide(dY,dZ1);

    return { dW, dB };
};

const train = (X,Y_lable,learning_rate,epoch)=>{

    for(let i = 0 ; i<epoch;i++){
        let {Y,Z1} = forwordPropagation(X,weights,bias);
        let {dW,dB} = backPropagation(Y,Y_lable,X);
        console.log(dW,dB);
        weights = subtract(weights,multiply(learning_rate,dW));
        bias = subtract(bias,multiply(learning_rate,dB));
    }

    return {weights,bias};
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

        for (let i =0;i<1;i++){
            let X = matrix([[parseInt(training_x1[i]),parseInt(training_x2[i])]]);
            let Y_lable = matrix([[parseInt(training_y[i])]]);
            let {weights,bias} = train(X,Y_lable,0.1,100);
            if(i%100 == 0){
                console.log(weights,bias);
                let result =0;
                for(let j=0;j<0;j++){
                    let X = matrix([[parseInt(testing_x1[j]),parseInt(testing_x2[j])]]);
                    let Y_lable = matrix([[parseInt(testing_y[j])]]);
                    let {Y,Z1} = forwordPropagation(X,weights,bias);
                    if(Y.get([0,0])>0.5 && Y_lable.get([0,0])== 1){
                        result++;
                    }
                    if(Y.get([0,0])<0.5 && Y_lable.get([0,0]) == 0){
                        result--;
                    }
                }
                console.log("Accuracy --- " , (result/testing_x1.length)*100 , "%");
            }
        }
    });

