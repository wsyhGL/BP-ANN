#pragma once
#ifndef CIRCLR_H
#define CIRCLR_H
#include "neuronLayer.h"
#include<vector>
using namespace std;
#define ACTIVATION_RESPONSE 0.7

#define BIAS                1
//神经网络
class bpNeuronoNet 
{
public:
	bpNeuronoNet(int numInputs, double learningRate);//构造函数
	~bpNeuronoNet();
public:
	inline double getError(void) { return mErrorSum; }//总误差
	bool training(const double inputs[], const double targets[]);//训练网络
	void process(const double inputs[], double* outputs[]);//识别数字
	void reset();//重置网络
	void addNeuronLayer(int numNeurons);//添加神经网络

private:
	inline double sigmoidActive(double activation, double response);//激活函数
	void updateNeuronLayer(neuronLayer& nl, const double inputs[]);
	inline double backActive(double x);//激活函数的导数
	void trainUpdate(const double inputs[], const double targets[]);
	void trainNeuronLayer(neuronLayer& nl, const double prevOutActivations[], double precOutErrors[]);

private:
	int mNumInputs;//神经网络的输入数目
	int mNumOutputs;//神经网络的输出数目
	int mNumHiddenLayers;//隐含层数目
	double mLearningRate;//神经网络的学习率
	double mErrorSum;//总误差
	vector<neuronLayer*>mNeuronLayers;
};
#endif

