const { createReadStream } = require('fs');
const csv = require('csv-parser');

const { exp } = require('mathjs');

let x1 = [];
let x2= [];

let y= [];

let weights = [1,1];
let bias = 1;

//Simple perceptron
const activation = (x) => {
    let sig = 1/(1+exp(-x));
    return sig;
};

const forwordPropagation=(X,weights,bias) => {
    let Z1 = weights[0]*X[0]+weights[1]*X[1]+bias;
    let Y = activation(Z1);
    return Y;
};

const backPropagation=(Y,Y_lable,X)=>{
    let E = (Y-Y_lable)**2;
    let dE_dY = 2*(Y-Y_lable);
    let dY_dZ1 = activation(Y)*(1-activation(Y));
    let dZ1_dW1 = X[0];
    let dZ1_dW2 = X[1];

    let dW = [dE_dY*dY_dZ1*dZ1_dW1,dE_dY*dY_dZ1*dZ1_dW2];
    let dB = dE_dY*dY_dZ1;

    return {dW,dB};
};

const train = (X,Y_lable,learning_rate,epoch)=>{

    for(let i = 0 ; i<epoch;i++){
        let Y = forwordPropagation(X,weights,bias);
        let {dW,dB} = backPropagation(Y,Y_lable,X);
        weights[0] = weights[0] - learning_rate*dW[0];
        weights[1] = weights[1] - learning_rate*dW[1];

        bias = bias - learning_rate*dB;
    }

    return {weights,bias};
}

let AND_Gate_X1 = [];
let AND_Gate_X2 = [];

let AND_Gate_Y = [];

for(let i = 0;i<5000;i++){
    AND_Gate_X1.push(Math.round(Math.random()));
    AND_Gate_X2.push(Math.round(Math.random()));
    AND_Gate_Y.push(AND_Gate_X1[i]*AND_Gate_X2[i]);
}


createReadStream('/home/banik/Desktop/Code/SimpleNN/DATA/XOR/perceptron_DataSet.csv')
    .pipe(csv())
    .on('data', (data) => {
        x1.push(data.x);
        x2.push(data.y);
        y.push(data.z);
    })
    .on('end', () => {
        console.log('CSV file successfully processed');

        /**
         * Testing
         */
        // x1 = AND_Gate_X1;
        // x2 = AND_Gate_X2;
        // y = AND_Gate_Y;

        let training_x1 = x1.splice(0, x1.length * 0.7);
        let training_x2 = x2.splice(0, x2.length * 0.7);
        let training_y= y.splice(0, y.length * 0.7);

        let testing_x1 = x1.splice(0, x1.length);
        let testing_x2 = x2.splice(0, x2.length);
        let testing_y = y.splice(0, y.length);
        for (let i =0;i<training_x1.length;i++){
            let X = [parseInt(training_x1[i]),parseInt(training_x2[i])];
            let Y_lable = parseInt(training_y[i]);
            let {weights,bias} = train(X,Y_lable,0.01,10);
            if(i%100 == 0){
                // console.log(weights,bias);
                let result =0;
                for(let j=0;j<testing_x1.length;j++){
                    let X = [parseInt(testing_x1[j]),parseInt(testing_x2[j])];
                    let Y_lable = parseInt(testing_y[j]);
                    let Y = forwordPropagation(X,weights,bias);
                    if(Y>0.5 && Y_lable == 1){
                        result++;
                    }
                    if(Y <0.5 && Y_lable == 0){
                        result++;
                    }
                }
                console.log(`Accuracy at ${i} --- ${(result/testing_x1.length)*100}%`);
                console.log(weights,bias);
            }
        }
    });

// setTimeout(()=>{
//     let result = 0;
//     for(let i =0;i<5000;i++){
//         let X = [AND_Gate_X1[i],AND_Gate_X2[i]];
//         let Y_lable = AND_Gate_Y[i];
//         let Y = forwordPropagation(X,weights,bias);
//         if(Y>0.5 && Y_lable == 1){
//             result++;
//         }
//         if(Y <0.5 && Y_lable == 0){
//             result++;
//         }
//     }
//     console.log(`Accuracy --- ${(result/5000)*100}%`);
// },5000);

