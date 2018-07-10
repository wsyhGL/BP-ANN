#pragma once
#ifndef CIRCLE_H
#define CIRCLE_H
struct neuronLayer
{
public:
	neuronLayer(int numNeurons, int numInputsPerNeuron);//���캯��
	neuronLayer(neuronLayer& nl);//�������캯��
	~neuronLayer();//��������

	void reset();//��ֵ����
public:
	int mNumInputsPerNeuron;//ÿ����ϸ����������Ŀ
	int mNumNeurons;//��ǰ�����ϸ����Ŀ
	double** mWeights;
	double* mOutActivations;//��ϸ�������ֵ
	double* mOutErrors;//���ֵ
};
#endif
