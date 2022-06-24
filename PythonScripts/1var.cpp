#include<iostream>
#include<immintrin.h>
#include<fstream>
#include<time.h>
#include<chrono>
#include <vector>
#include <fstream>
#include<string>

using namespace std;

const int picSize = 28 * 28;
const string trainDataPath = "D:\\Study\\special\\Project1\\Project1\\train.txt";
const string testDataPath = "D:\\Study\\special\\Project1\\Project1\\test.txt";
const string weightsPath = "D:\\Study\\special\\Project1\\Project1\\weights.txt";

class network {
public:
	int layersN;
	int* arch;		//архитектура
	string* activationFunctions;
	long long neuronsNum = 0;
	long long weightsNum = 0;
	double* values;		//массив значений нейронов
	double* errors;		//массив ошибок
	double* weights;	//массив значений весов

	//конструктор
	network(int lN, int* tempArch, string* activationFunction) {
		layersN = lN;
		arch = new int[lN];
		activationFunctions = new string[lN];
		for (int i = 0; i < lN; i++) {
			arch[i] = tempArch[i];
			activationFunctions[i] = activationFunction[i];
			neuronsNum += arch[i];

			//считаем количество связей
			if (i > 0) {
				weightsNum += arch[i] * arch[i - 1];
			}
		}
		weights = new double[weightsNum];
		values = new double[neuronsNum];
		errors = new double[neuronsNum];
	}

	//умножаем матрицы
	void gemm(int M, int N, int K, const double* A, const double* B, double* C) {
		for (int i = 0; i < M; i++) {
			double* c = C + i * N;
			for (int j = 0; j < N; j++) {
				c[j] = 0;
			}
			for (int k = 0; k < K; k++) {
				const double* b = B + k * N;
				double a = A[i * K + k];
				for (int j = 0; j < N; j++) {
					c[j] += a * b[j];
				}
			}
		}
	}

	void gemmSumm(int M, int N, int K, const double* A, const double* B, double* C) {
		for (int i = 0; i < M; i++) {
			double* c = C + i * N;
			for (int k = 0; k < K; k++) {
				const double* b = B + k * N;
				double a = A[i * K + k];
				for (int j = 0; j < N; j++) {
					c[j] += a * b[j];
				}
			}
		}
	}

	void New() {
		long long counter = 0;
		for (int i = 0; i < layersN - 1; i++) {
			for (int j = 0; j < arch[i] * arch[i + 1]; j++) {
				weights[counter] = (double(rand() % 101) / 100) / arch[i + 1];
				counter++;
			}
		}
	}

	//функция активации
	void act(int valuesC, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				values[i] = (1 / (1 + pow(2.71828, -values[i])));
			}
		}
		if (activationFunctions[layer] == "relu") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				if (values[i] < 0) {
					values[i] = 0;
				}
				else {
					values[i] *= 0.01;
				}
			}
		}
	}

	//производная от функции активации
	void pro(double* value, int ecounter, int layer) {
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
	}

	void forwardFeed(double* input_data) {
		for (int i = 0; i < arch[0]; i++) {
			values[i] = input_data[i];

		}



		long long valuesC = 0;
		long long weightsC = 0;
		for (int i = 0; i < layersN - 1; i++) {
			double* a = values + valuesC;		//значения на нейронах
			double* b = weights + weightsC;		//веса 
			double* c = values + valuesC + arch[i];		//значения следущего слоя
			gemm(1, arch[i + 1], arch[i], a, b, c);		//умножаем матрицы

			valuesC += arch[i];
			act(valuesC, i + 1);

			weightsC += arch[i] * arch[i + 1];

		}
	}

	void backPropogation(double* rightResults, float lr) {
		//вычисляем ошибку
		int h = neuronsNum - arch[layersN - 1];
		for (int i = 0; i < arch[layersN - 1]; i++) {
			errors[i + h] = rightResults[i] - values[i + h];
		}

		long long weightCounter = weightsNum;
		long long errorCounter = neuronsNum;
		long long counter = neuronsNum - arch[layersN - 1];
		for (int i = layersN - 1; i > 0; i--) {
			errorCounter -= arch[i];
			weightCounter -= arch[i] * arch[i - 1];
			counter -= arch[i - 1];
			double* a = errors + errorCounter;
			double* b = weights + weightCounter;
			double* c = errors + counter;
			gemm(1, arch[i - 1], arch[i], a, b, c);
		}


		//обновляем веса
		long long vcounter = neuronsNum - arch[layersN - 1];
		weightCounter = weightsNum;
		errorCounter = neuronsNum;
		for (int i = layersN - 1; i > 0; i--) {
			errorCounter -= arch[i];
			vcounter -= arch[i - 1];
			weightCounter -= arch[i] * arch[i - 1];
			double* b = new double[arch[i]];
			for (int j = 0; j < arch[i]; j++) {
				b[j] = errors[errorCounter + j] * lr;
			}

			pro(b, errorCounter, i);
			double* a = values + vcounter;
			double* c = weights + weightCounter;

			gemmSumm(arch[i - 1], arch[i], 1, a, b, c);

			delete[] b;
		}
	}

	void getPrediction(float* result) {
		long long h = neuronsNum - arch[layersN - 1];
		
		for (int i = 0; i < arch[layersN - 1]; i++) {
			result[i] = values[h + i];
			
			//cout << "Result[" << i << "]: " << result[i] << endl;
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
	float info[picSize];
	int rresult;
};

int main() {

	srand(time(0));
	setlocale(LC_ALL, "Russian");
	const int N = 4;
	int size[N] = { picSize, 16, 16, 10 };
	string AF[N] = { "relu", "sigmoid", "relu", "relu" };
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
		
		while (percentRight <= 75.0) {
			fin.open(trainDataPath);

			for (int i = 0; i < 1000; i++) {
				double input[picSize];

				for (int j = 0; j < picSize; j++) {
					fin >> input[j];
					/*cout << input[j] << " ";
					if (j % 28 == 0 && j != 0) {
						cout << "\n";
					}
					*/
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

				double rightResults[10];
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
					n.backPropogation(rightResults, 0.25);
				}
			}


			fin.close();

		}
		n.SaveWeights(weightsPath);
		cout << "тестовые данные : \n";
		fin.open(testDataPath);
		for (int i = 1; i < 11; i++) {
			double input[picSize];

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
		for (int i = 0; i < n.weightsNum; i++) {
			cout << n.weights[i] << " ";
		}
		cout << endl;
		fin.open(testDataPath);
		for (int i = 1; i < 11; i++) {
			double input[picSize];

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
};
