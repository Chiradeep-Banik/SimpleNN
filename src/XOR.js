const { createReadStream } = require('fs');
const csv = require('csv-parser');

const { matrix,
    exp,
    multiply,
    add,
    divide,
    subtract, 
    transpose} = require('mathjs');

let x1 = [];
let x2= [];

let y= [];

let weights = matrix([[1],[1]]);
let bias = matrix([[1]]);

//Simple perceptron
const activation = (x) => {
    let sig =divide(1, add(1,exp(multiply(-1,x))));
    if(sig>=0.5){
        return matrix([[1]]);
    }else{
        return matrix([[0]]);
    }
};

const forwordPropagation=(X,weights,bias) => {
    let Z1 = add(multiply(X,weights),bias);
    let Y = activation(Z1);
    return {Y,Z1};
};

const backPropagation=(Y,Y_lable,X)=>{
    let dY = subtract(Y,Y_lable);
    let dW = multiply(transpose(X),dY);
    let dB = multiply(dY,bias);

    // console.log(dW,dW);
    return { dW, dB };
};

const train = (X,Y_lable,learning_rate,epoch)=>{

    for(let i = 0 ; i<epoch;i++){
        let {Y,Z1} = forwordPropagation(X,weights,bias);
        let {dW,dB} = backPropagation(Y,Y_lable,X);
        weights = add(weights,multiply(learning_rate,dW));
        bias = add(bias,multiply(learning_rate,dB));
    }

    return {weights,bias};
}


// let X = matrix([[1,1]]);
// let Y_lable = matrix([[1]]);

// train(X,Y_lable,0.01,100);
// console.log(weights,bias);

createReadStream('/home/banik/Desktop/Code/SimpleNN/DATA/XOR/other.csv')
    .pipe(csv())
    .on('data', (data) => {
        x1.push(data.x);
        x2.push(data.y);
        y.push(data.z);
        console.log(data);
    })
    .on('end', () => {
        console.log('CSV file successfully processed');
        let training_x1 = x1.splice(0, x1.length * 0.7);
        let training_x2 = x2.splice(0, x2.length * 0.7);
        let training_y= y.splice(0, y.length * 0.7);

        let testing_x1 = x1.splice(0, x1.length);
        let testing_x2 = x2.splice(0, x2.length);
        let testing_y = y.splice(0, y.length);
        console.log(training_x1[1],training_x2[1],training_y[1]);
        for (let i =0;i<training_x1.length;i++){
            let X = matrix([[parseInt(training_x1[i]),parseInt(training_x2[i])]]);
            let Y_lable = matrix([[parseInt(training_y[i])]]);
            let {weights,bias} = train(X,Y_lable,0.01,10);
            if(i%100 == 0){
                // console.log(weights,bias);
                let result =0;
                for(let j=0;j<testing_x1.length;j++){
                    let X = matrix([[parseInt(testing_x1[j]),parseInt(testing_x2[j])]]);
                    let Y_lable = matrix([[parseInt(testing_y[j])]]);
                    let {Y,Z1} = forwordPropagation(X,weights,bias);
                    if(Y.get([0,0])>0.5 && Y_lable.get([0,0])== 1){
                        result++;
                    }
                    if(Y.get([0,0])<0.5 && Y_lable.get([0,0]) == 0){
                        result++;
                    }
                }
                console.log(`Accuracy at ${i} --- ${(result/testing_x1.length)*100}%`);
                console.log(weights,bias);
            }
        }
    });

