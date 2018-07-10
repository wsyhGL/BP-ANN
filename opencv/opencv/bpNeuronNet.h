#pragma once
#ifndef CIRCLR_H
#define CIRCLR_H
#include "neuronLayer.h"
#include<vector>
using namespace std;
#define ACTIVATION_RESPONSE 0.7

#define BIAS                1
//������
class bpNeuronoNet 
{
public:
	bpNeuronoNet(int numInputs, double learningRate);//���캯��
	~bpNeuronoNet();
public:
	inline double getError(void) { return mErrorSum; }//�����
	bool training(const double inputs[], const double targets[]);//ѵ������
	void process(const double inputs[], double* outputs[]);//ʶ������
	void reset();//��������
	void addNeuronLayer(int numNeurons);//���������

private:
	inline double sigmoidActive(double activation, double response);//�����
	void updateNeuronLayer(neuronLayer& nl, const double inputs[]);
	inline double backActive(double x);//������ĵ���
	void trainUpdate(const double inputs[], const double targets[]);
	void trainNeuronLayer(neuronLayer& nl, const double prevOutActivations[], double precOutErrors[]);

private:
	int mNumInputs;//�������������Ŀ
	int mNumOutputs;//������������Ŀ
	int mNumHiddenLayers;//��������Ŀ
	double mLearningRate;//�������ѧϰ��
	double mErrorSum;//�����
	vector<neuronLayer*>mNeuronLayers;
};
#endif

