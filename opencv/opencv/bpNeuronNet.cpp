#include "bpNeuronNet.h"
#include "math.h"

bpNeuronoNet::bpNeuronoNet(int numInputs, double learningRate)
	:mNumInputs(numInputs),
	mNumOutputs(0),
	mNumHiddenLayers(0),
	mLearningRate(learningRate),
	mErrorSum(9999)
{
}

bpNeuronoNet::~bpNeuronoNet()
{

	for (size_t i = 0; i < mNeuronLayers.size(); i++)
	{
		if (mNeuronLayers[i])
		{
			delete mNeuronLayers[i];
		}
	}

}

void bpNeuronoNet::reset()
{
	//for each layer
	for (int i = 0; i < mNumHiddenLayers + 1; ++i)
	{
		mNeuronLayers[i]->reset();
	}

	mErrorSum = 9999;
}

void bpNeuronoNet::addNeuronLayer(int numNeurons)
{
	int numInputsPerNeuron = (mNeuronLayers.size() > 0) ? mNeuronLayers[mNumHiddenLayers]->mNumNeurons : mNumInputs;

	/** create a neuron layer */
	mNeuronLayers.push_back(new neuronLayer(numNeurons, numInputsPerNeuron));

	/** calculate the count of hidden layers */
	mNumHiddenLayers = (mNeuronLayers.size() > 0) ? (mNeuronLayers.size() - 1) : 0;
}

//�����
double bpNeuronoNet::sigmoidActive(double activation, double response) 
{
	return (1.0 / (1.0 + exp(-activation* response)));
}
//����һ��������㣬�������
void bpNeuronoNet::updateNeuronLayer(neuronLayer& nl, const double inputs[])
{
	int numNeurons = nl.mNumNeurons;//��ǰ���ϸ����
	int numInputsPerNeuron = nl.mNumInputsPerNeuron;//�������
	double* curOutActivations = nl.mOutActivations;//�������
	for (int i = 0; i < numNeurons; i++) 
	{
		double* curWeights = nl.mWeights[i];//��ϸ������Ȩ������
		double net = 0;
		int k;
		for (k = 0; k < numInputsPerNeuron; k++) 
		{
			net += curWeights[k] * inputs[k];
		}
		net += curWeights[k] * BIAS;
        //�õ���ǰ��ϸ�������ֵ
		curOutActivations[i] = sigmoidActive(net, ACTIVATION_RESPONSE);
	}
}

void bpNeuronoNet::process(const double inputs[], double* outputs[])
{
	//������
	for (int i = 0; i < mNumHiddenLayers + 1; i++) 
	{
		updateNeuronLayer(*mNeuronLayers[i], inputs);
		inputs = mNeuronLayers[i]->mOutActivations;
	}
	//��ȡ��ϸ���������
	*outputs = mNeuronLayers[mNumHiddenLayers]->mOutActivations;
}
double bpNeuronoNet::backActive(double x)
{
	return x*(1 - x);
}
//��ѵ����ģʽ��������ĺ���
void bpNeuronoNet::trainUpdate(const double inputs[], const double targets[])
{
	for (int i = 0; i < mNumHiddenLayers + 1; i++)
	{
		updateNeuronLayer(*mNeuronLayers[i], inputs);
		inputs = mNeuronLayers[i]->mOutActivations;
	}
	//��ȡ����������
	neuronLayer& outLayer = *mNeuronLayers[mNumHiddenLayers];
	double *outActivations = outLayer.mOutActivations;//�����ϸ�����������
	double *outErrors = outLayer.mOutErrors;//�������ϸ������������
	int numNeurons = outLayer.mNumNeurons;//��������ϸ������

	mErrorSum = 0;//���������
	for (int i = 0; i < numNeurons; i++)
	{
		double err = targets[i] - outActivations[i];
		outErrors[i] = err;
		mErrorSum += err*err;
	}
}
void bpNeuronoNet::trainNeuronLayer(neuronLayer& nl, const double precOutActivations[], double prevOutErrors[])
{
	int numNeurons = nl.mNumNeurons;
	int numInputsPerNeuron = nl.mNumInputsPerNeuron;
	double *curOutErrors = nl.mOutErrors;
	double *curOutActivations = nl.mOutActivations;

	for (int i = 0; i < numNeurons; i++) 
	{
		double *curWeights = nl.mWeights[i];
		double coi = curOutActivations[i];
		double err = curOutErrors[i] * backActive(coi);
		int w;
		//������ǰϸ����Ȩ��
		for (w = 0; w < numInputsPerNeuron; w++)
		{
			if (prevOutErrors)
			{
				prevOutErrors[w] += curWeights[w] * err;
			}
			//����Ȩ��
			curWeights[w] += err*mLearningRate*precOutActivations[w];
		}
		curWeights[w] += err*mLearningRate*BIAS;

	}
}
bool bpNeuronoNet::training(const double inputs[], const double targets[])
{
	const double *prevOutActivations = NULL;
	double *prevOutErrors = NULL;
	trainUpdate(inputs, targets);//��ѵ��ģʽ��������
	for (int i = mNumHiddenLayers; i >= 0; i--) 
	{
		neuronLayer& curLayer = *mNeuronLayers[i];//��ȡ��i����ϸ����
		if (i > 0)
		{
			neuronLayer& prev = *mNeuronLayers[(i - 1)];
			prevOutActivations = prev.mOutActivations;
			prevOutErrors = prev.mOutErrors;
			memset(prevOutErrors, 0, prev.mNumNeurons * sizeof(double));
		}
		else
		{
			prevOutActivations = inputs;
			prevOutErrors = NULL;

		}
		trainNeuronLayer(curLayer, prevOutActivations, prevOutErrors);
	}
	return true;
}
