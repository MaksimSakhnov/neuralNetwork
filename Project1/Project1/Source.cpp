#include<iostream>
#include<immintrin.h>
#include<fstream>
#include<time.h>
#include<chrono>
using namespace std;
const int picSize = 28 * 28;
const string trainDataPath = "D:\\Study\\special\\Project1\\Project1\\train.txt";
const string testDataPath = "D:\\Study\\special\\Project1\\Project1\\test.txt";
const string weightsPath = "D:\\Study\\special\\Project1\\Project1\\weights.txt";

class network {
public:
	int layersN;
	int* arch;
	string* activationFunctions;
	long long neuronsNum = 0;
	long long weightsNum = 0;
	float* values;	//values of neurons
	float* errors;
	float* weights; // weights of neurons
	network(int lN, int* Arch, string* AF) {
		layersN = lN;
		arch = new int[lN];
		activationFunctions = new string[lN];
		for (int i = 0; i < lN; i++) {
			arch[i] = Arch[i];
			activationFunctions[i] = AF[i];
			neuronsNum += arch[i];
			if (i > 0) {
				weightsNum += arch[i] * arch[i - 1];
			}
		}
		weights = new float[weightsNum];
		values = new float[neuronsNum];
		errors = new float[neuronsNum];
	}

	void gemm(int M, int N, int K, const float* A, const float* B, float* C)
	{
		for (int i = 0; i < M; ++i)
		{
			float* c = C + i * N;
			for (int j = 0; j < N; ++j)
				c[j] = 0;
			for (int k = 0; k < K; ++k)
			{
				const float* b = B + k * N;
				float a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += a * b[j];
			}
		}
	}
	void gemmSum(int M, int N, int K, const float* A, const float* B, float* C)
	{
		for (int i = 0; i < M; ++i)
		{
			float* c = C + i * N;
			for (int k = 0; k < K; ++k)
			{
				const float* b = B + k * N;
				float a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += a * b[j];
			}
		}
	}

	void New() {
		long long counter = 0;
		for (int i = 0; i < layersN - 1; i++) {
			for (int j = 0; j < arch[i] * arch[i + 1]; j++) {
				weights[counter] = (double(rand() % 101) / 100.0) / arch[i + 1];
				counter++;
			}
		}
	}

	void act(int valuesC, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				values[i] = (1 / (1 + pow(2.71828, -values[i])));
			}

		}
		if (activationFunctions[layer] == "relu") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				if (values[i] < 0) values[i] *= 0.01;
			}
		}
		if (activationFunctions[layer] == "softmax") {
			float zn = 0.0;
			for (int i = 0; i < arch[layer]; i++) {
				zn += pow((2.71), values[valuesC + i]);
			}
			for (int i = 0; i < arch[layer]; i++) {
				values[valuesC + i] = pow(2.71, values[valuesC + i]) / zn;
			}
		}
	}

	void pro(float* value, int ecounter, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = 0; i < arch[layer]; i++) {
				values[ecounter + i] = values[ecounter + i] * (1.0 - values[ecounter + i]);
				value[i] *= values[ecounter + i];
			}
		}

		if (activationFunctions[layer] == "relu") {
			for (int i = 0; i < arch[layer]; i++) {
				if (values[i + ecounter] < 0) values[i] = 0.01;
				else values[i + ecounter] = 1.0;
				value[i] *= values[ecounter + i];
			}
		}

		if (activationFunctions[layer] == "softmax") {
			for (int i = 0; i < arch[layer]; i++) {
				value[i] *= values[ecounter + i] * (1.0 - values[ecounter + i]);
			}
		}
	}

	void forwardFeed(float* input_data) {
		for (int i = 0; i < arch[0]; i++) {
			values[i] = input_data[i];
		}

		long long valuesC = 0;
		long long weightsC = 0;
		for (int i = 0; i < layersN - 1; i++) {
			float* a = values + valuesC;
			float* b = weights + weightsC;
			float* c = values + valuesC + arch[i];
			gemm(1, arch[i + 1], arch[i], a, b, c);


			//for (int j = valuesC; j < valuesC + arch[i + 1]; j++) {
			valuesC += arch[i];
			act(valuesC, i + 1);



			weightsC += arch[i] * arch[i + 1];

		}

	}

	void getPrediction(float* result) {
		long long h = neuronsNum - arch[layersN - 1];
		float sum = 0;
		for (int i = 0; i < arch[layersN - 1]; i++) {
			result[i] = values[h + i];
			sum += result[i];
			//cout << "Result[" << i << "]: " << result[i] << endl;
		}
	}

	void backPropogation(float* rightResults, float lr) {
		//Сначала вычисление ошибок
		int h = neuronsNum - arch[layersN - 1];
		for (int i = 0; i < arch[layersN - 1]; i++) {
			errors[i + h] = rightResults[i] - values[i + h];
		}
		long long wcounter = weightsNum;
		long long ecounter = neuronsNum;
		long long counter = neuronsNum - arch[layersN - 1];
		for (int i = layersN - 1; i > 0; i--) {
			ecounter -= arch[i];
			wcounter -= arch[i] * arch[i - 1];
			counter -= arch[i - 1];
			float* a = errors + ecounter;
			float* b = weights + wcounter;
			float* c = errors + counter;
			gemm(1, arch[i - 1], arch[i], a, b, c);

		}

		//Потом обновление весов:
		long long vcounter = neuronsNum - arch[layersN - 1];
		wcounter = weightsNum;
		ecounter = neuronsNum;
		for (int i = layersN - 1; i > 0; i--) {
			ecounter -= arch[i];
			vcounter -= arch[i - 1];
			wcounter -= arch[i] * arch[i - 1];
			float* b = new float[arch[i]];
			for (int j = 0; j < arch[i]; j++) {
				b[j] = errors[ecounter + j] /*pro(values[ecounter + j], i)*/ * lr;
			}
			pro(b, ecounter, i);
			float* a = values + vcounter;
			float* c = weights + wcounter;

			gemmSum(arch[i - 1], arch[i], 1, a, b, c);

			delete[] b;
		}

	}

	void SaveWeights(string filename) {
		ofstream fout;
		fout.close();
		fout.open(filename);
		for (int i = 0; i < weightsNum; i++) {
			fout << weights[i]<< " ";
		}
		fout.close();
	}

	void LoadWeights(string filename) {
		ifstream fin;
		fin.open(filename);

		float temp;
		for (int i = 0; i < weightsNum; i++) {
			fin >> temp;

			weights[i] = temp;
		}
	}
};

struct data_one {
	float info[28 * 28];
	int rresult;
};

int main() {
	srand(time(0));
	setlocale(LC_ALL, "Russian");
	const int N = 4;
	int size[N] = { picSize, 400, 60, 10 };
	string AF[N] = { "relu", "sigmoid", "sigmoid", "softmax" };
	network n(N, size, AF);
	n.New();
	int counter = 0;
	cout << "обучать - 0\n не обучать - 1\n";
	int var;
	cin >> var;



	if (var == 0) {
		ifstream fin;
		string test;

		double percentRight = 0;
		double countRightResults = 0;
		double testCounter = 0;

		while (percentRight<70.0) {
			fin.open(trainDataPath);

			for (int i = 0; i < 10001; i++) {
				if (percentRight > 70.0) break;
				float input[picSize];

				for (int j = 0; j < picSize; j++) {
					fin >> input[j];
					//cout << input[j] << " ";
					//if (j % 28 == 0 && j != 0) {
						//cout << "\n";
					//}

				}

				string rightAnswer;
				fin >> rightAnswer;
				n.forwardFeed(input);
				float result[10];
				n.getPrediction(result);

				float maxV = -1;
				int maxNumber = -1;

				for (int j = 0; j < 10; j++) {
					if (maxV <= result[j]) {
						maxV = result[j];
						maxNumber = j;
					}
				}
				cout << endl;
				cout << "трениpовочные данные: " << testCounter << " Ответ: " << maxNumber << " Правильный ответ: " << rightAnswer;

				float rightResults[10];
				for (int j = 0; j < 10; j++) {
					rightResults[j] = 0;
				}
				cout << " check: " << maxV;

				bool flag = 0;
				if (rightAnswer == "zero") {
					if (maxNumber == 0) {
						countRightResults++;
						flag = 1;

					}
					rightResults[0] = 1;

				}
				else if (rightAnswer == "one") {
					if (maxNumber == 1) {
						countRightResults++;
						flag = 1;

					}
					rightResults[1] = 1;
				}
				else if (rightAnswer == "two") {
					if (maxNumber == 2) {
						countRightResults++;
						flag = 1;

					}
					rightResults[2] = 1;

				}
				else if (rightAnswer == "three") {
					if (maxNumber == 3) {
						countRightResults++;
						flag = 1;

					}
					rightResults[3] = 1;

				}
				else if (rightAnswer == "four") {
					if (maxNumber == 4) {
						countRightResults++;
						flag = 1;

					}
					rightResults[4] = 1;

				}
				else if (rightAnswer == "five") {
					if (maxNumber == 5) {
						countRightResults++;
						flag = 1;

					}
					rightResults[5] = 1;

				}
				else if (rightAnswer == "six") {
					if (maxNumber == 6) {
						countRightResults++;
						flag = 1;

					}
					rightResults[6] = 1;

				}
				else if (rightAnswer == "seven") {
					if (maxNumber == 7) {
						countRightResults++;
						flag = 1;

					}
					rightResults[7] = 1;

				}
				else if (rightAnswer == "eight") {
					if (maxNumber == 8) {
						countRightResults++;
						flag = 1;

					}
					rightResults[8] = 1;
				}
				else if (rightAnswer == "nine") {
					if (maxNumber == 9) {
						countRightResults++;
						flag = 1;

					}
					rightResults[9] = 1;

				}
				testCounter++;
				percentRight = countRightResults / testCounter;
				percentRight *= 100;

				cout << " количество правильных: " << countRightResults << " процент правильных ответов: " << percentRight << endl;
				if (!flag) {
					n.backPropogation(rightResults, 0.5);
				}
			}


			fin.close();

		}
		n.SaveWeights(weightsPath);
		cout << "тестовые данные : \n";
		fin.open(testDataPath);
		for (int i = 1; i < 11; i++) {
			float input[picSize];

			for (int j = 0; j < picSize; j++) {
				fin >> input[j];
				cout << input[j] << " ";
				if (j % 28 == 0 && j != 0) {
					cout << "\n";
				}
			}

			n.forwardFeed(input);
			float result[10];
			n.getPrediction(result);

			float maxV = -1;
			int maxNumber = -1;

			for (int j = 0; j < 10; j++) {
				if (maxV <= result[j]) {
					maxV = result[j];
					maxNumber = j;
				}
			}
			cout << endl;
			cout << "тестовые данные: " << i << " Ответ: " << maxNumber << endl;
			//delete[] result;

		}

	}
	else if (var == 1) {

	n.LoadWeights(weightsPath);
	ifstream fin;
	//for (int i = 0; i < n.weightsNum; i++) {
		//cout << n.weights[i] << " ";
	//}
	cout << endl;
	fin.open(testDataPath);
	for (int i = 1; i < 21; i++) {
		float input[picSize];

		for (int j = 0; j < picSize; j++) {
			fin >> input[j];
			cout << input[j] << " ";
			if (j % 28 == 0 && j != 0) {
				cout << "\n";
			}
		}

		n.forwardFeed(input);
		float result[10];
		n.getPrediction(result);

		float maxV = -1;
		int maxNumber = -1;

		for (int j = 0; j < 10; j++) {
			if (maxV <= result[j]) {
				maxV = result[j];
				maxNumber = j;
			}
		}
		cout << endl;
		cout << "тестовые данные: " << i << " Ответ: " << maxNumber << endl;
		//delete[] result;

	}

	}


	return 0;
}