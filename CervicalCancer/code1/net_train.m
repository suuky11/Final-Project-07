clc;
clear all;
close all;
% warning off all;
% function y = net_train(net1,A,x)
%% Load the data

load data


%% specify the input and the targets

% A = [data1,data2,data3,data4,data5,data6];
A = data;
x1=ones(1,20);
x2=ones(1,20)*2;
x3=ones(1,20)*3;

x = [x1 x2 x3];

%% Create a neural network

net1 = newff(minmax(A),[50 30 1],{'logsig','logsig','purelin'},'trainrp');
net1.trainParam.show = 1000;
net1.trainParam.lr = 0.0001;
net1.trainParam.epochs = 7000;
net1.trainParam.goal = 1e-10;

%% Train the neural network using the input,target and the created network

[net1] = train(net1,A,x);

%% Save the network

save net1 net1

%% Simulate the network for a particular input

y = round(sim(net1,A))







