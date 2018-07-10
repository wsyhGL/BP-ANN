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

//激活函数
double bpNeuronoNet::sigmoidActive(double activation, double response) 
{
	return (1.0 / (1.0 + exp(-activation* response)));
}
//更新一个神经网络层，计算输出
void bpNeuronoNet::updateNeuronLayer(neuronLayer& nl, const double inputs[])
{
	int numNeurons = nl.mNumNeurons;//当前层的细胞数
	int numInputsPerNeuron = nl.mNumInputsPerNeuron;//输入个数
	double* curOutActivations = nl.mOutActivations;//输出数组
	for (int i = 0; i < numNeurons; i++) 
	{
		double* curWeights = nl.mWeights[i];//神经细胞输入权重数组
		double net = 0;
		int k;
		for (k = 0; k < numInputsPerNeuron; k++) 
		{
			net += curWeights[k] * inputs[k];
		}
		net += curWeights[k] * BIAS;
        //得到当前神经细胞的输出值
		curOutActivations[i] = sigmoidActive(net, ACTIVATION_RESPONSE);
	}
}

void bpNeuronoNet::process(const double inputs[], double* outputs[])
{
	//逐层更新
	for (int i = 0; i < mNumHiddenLayers + 1; i++) 
	{
		updateNeuronLayer(*mNeuronLayers[i], inputs);
		inputs = mNeuronLayers[i]->mOutActivations;
	}
	//获取神经细胞输出数组
	*outputs = mNeuronLayers[mNumHiddenLayers]->mOutActivations;
}
double bpNeuronoNet::backActive(double x)
{
	return x*(1 - x);
}
//以训练的模式更新网络的函数
void bpNeuronoNet::trainUpdate(const double inputs[], const double targets[])
{
	for (int i = 0; i < mNumHiddenLayers + 1; i++)
	{
		updateNeuronLayer(*mNeuronLayers[i], inputs);
		inputs = mNeuronLayers[i]->mOutActivations;
	}
	//获取网络的输出层
	neuronLayer& outLayer = *mNeuronLayers[mNumHiddenLayers];
	double *outActivations = outLayer.mOutActivations;//输出层细胞的输出数字
	double *outErrors = outLayer.mOutErrors;//输出层神经细胞的输出误差数
	int numNeurons = outLayer.mNumNeurons;//输出层的神经细胞数量

	mErrorSum = 0;//重置总误差
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
		//遍历当前细胞的权重
		for (w = 0; w < numInputsPerNeuron; w++)
		{
			if (prevOutErrors)
			{
				prevOutErrors[w] += curWeights[w] * err;
			}
			//更新权重
			curWeights[w] += err*mLearningRate*precOutActivations[w];
		}
		curWeights[w] += err*mLearningRate*BIAS;

	}
}
bool bpNeuronoNet::training(const double inputs[], const double targets[])
{
	const double *prevOutActivations = NULL;
	double *prevOutErrors = NULL;
	trainUpdate(inputs, targets);//以训练模式更新网络
	for (int i = mNumHiddenLayers; i >= 0; i--) 
	{
		neuronLayer& curLayer = *mNeuronLayers[i];//获取第i层神经细胞层
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
