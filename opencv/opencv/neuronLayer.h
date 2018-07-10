#pragma once
#ifndef CIRCLE_H
#define CIRCLE_H
struct neuronLayer
{
public:
	neuronLayer(int numNeurons, int numInputsPerNeuron);//构造函数
	neuronLayer(neuronLayer& nl);//拷贝构造函数
	~neuronLayer();//析构函数

	void reset();//冲值函数
public:
	int mNumInputsPerNeuron;//每个神经细胞的输入数目
	int mNumNeurons;//当前层的神经细胞数目
	double** mWeights;
	double* mOutActivations;//神经细胞的输出值
	double* mOutErrors;//误差值
};
#endif
