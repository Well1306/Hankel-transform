// MKL_FFT_SAMPLE_FROM_DOCUMENTATION.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <math.h>
#include "mkl.h"
#include "mkl_df_types.h"
#include <mkl_service.h>
#include <mkl_dfti.h>
// FFT.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <fstream>
#include <math.h>
#include "mkl.h"



typedef double (*Func) (double x);
const double beta = 0.01;
double PI = 3.14159265358979323846;

double expBeta(double x)
{
	return exp(-beta * abs(x));
}

double input_func(double x)
{
	return x * exp(-beta * pow(x, 2));
}

double output_func(double x)
{
	return exp(-pow(x, 2) / (4 * beta)) / (2 * beta);
}

double FFT_expBeta(double w)
{
	return 2 * beta / (beta * beta + 4 * PI * PI * w * w);
}

void FFT_expBeta_estimation(double epsF, double epsFFT, double& FNykuist, double& Period, double& deltaX, int& N)
{
	Period = -log(epsF) / beta;
	FNykuist = sqrt(2 * beta / epsFFT - beta * beta) / (2 * PI);
	deltaX = 0.5 / FNykuist;
	N = 2 * Period / deltaX;
}

double sum_sampling(double x, double deltaX, int N)
{
	double res = 0;
	for (int j = -N; j < N; j++)
	{
		double _x = PI * (x - j * deltaX) / deltaX;
		if (abs(_x) < 1.0e-8) res += expBeta(j * deltaX);
		else res += expBeta(j * deltaX) * sin(_x) / _x;
	}
	return res;
}

void expBeta_by_sampling(double epsF, double epsFFT)
{
	double Period = 0;
	double FNykuist = 0;
	double deltaX = 0;
	int N = 0;
	FFT_expBeta_estimation(epsF, epsFFT, FNykuist, Period, deltaX, N);

	double xs[]{ -500, -100, -50, -10, -1, 0, 1, 10, 50, 100, 200, 500 };

	for (int j = 0; j < 12; j++)
	{
		double est = sum_sampling(xs[j], deltaX, N);
		double exact = expBeta(xs[j]);
		std::cout << "x = " << xs[j] << " est = " << est << " exact = " << exact << " diff = " << abs(exact - est) << std::endl;
	}

}


int main() {

	double epsF = 1.0e-3;
	double epsFFT = 1.0e-3;
	double FNykuist = 0;
	double Period = 0;
	double deltax = 0;
	int _N = 0;

	FFT_expBeta_estimation(epsF, epsFFT, FNykuist, Period, deltax, _N);
	std::cout << "beta = " << beta << std::endl;
	std::cout << "epsF = " << epsF << " Period = " << Period << std::endl;
	std::cout << "epsFFT = " << epsFFT << " FNykuist = " << FNykuist << " deltax = " << deltax << std::endl;
	std::cout << " _N = " << _N << std::endl;
	expBeta_by_sampling(epsF, epsFFT);

	std::ofstream  fs("test.txt");

	fs << "beta = " << beta << std::endl;
	fs << "epsF = " << epsF << " Period = " << Period << std::endl;
	fs << "epsFFT = " << epsFFT << " FNykuist = " << FNykuist << " deltax = " << deltax << std::endl;
	fs << " _N = " << _N << std::endl;

	int err = 1;
	//Ввод шага дискретизации
	const double delta = 0.5;

	//Ввод границ интегрирования
	//double N = 10;
	double L = 700;

	//Получение массива значений функции f
	const int count = 2 * L / delta;
	double* nodes = new double[count];
	for (int i = 0; i < count; i++)
	{
		nodes[i] =(double) - L + delta * i;
	}

	std::string s;
	err = 1;

	int ngrid[] = { count };
	//const int howmany = 1;
	const size_t distanse = count;

	//FFT для входной функции
	double* input_array = (double*)mkl_malloc(count * sizeof(double), 64);
	double* input_fft = (double*)mkl_malloc(count * sizeof(double), 64);
	double* input_ifft = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
	{
		input_array[i] = input_func(nodes[i]);
	}

	DFTI_DESCRIPTOR_HANDLE inputF;

	MKL_LONG status = DftiCreateDescriptor(&inputF, DFTI_DOUBLE, DFTI_REAL,
		1, (MKL_LONG)count);
	std::string error_message;

	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(inputF, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(inputF);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(inputF, input_array, input_fft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	status = DftiComputeBackward(inputF, input_fft, input_ifft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeBackward: " << error_message << "\t " << status << "\n";


	//FFT для выходной функции
	double* output_array = (double*)mkl_malloc(count * sizeof(double), 64);
	double* output_fft = (double*)mkl_malloc(count * sizeof(double), 64);
	double* output_ifft = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
	{
		output_array[i] = output_func(nodes[i]);
	}

	DFTI_DESCRIPTOR_HANDLE outputF;
	status = DftiCreateDescriptor(&outputF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(outputF, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(outputF);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(outputF, output_array, output_fft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	status = DftiComputeBackward(outputF, output_fft, output_ifft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeBackward: " << error_message << "\t " << status << "\n";

	//Получаем преобразование Фурье Функции Бесселя по теореме о свертке
	double* bessel_fft = (double*)mkl_malloc(count * sizeof(double), 64);
	for (int i = 0; i < count; i++)
		bessel_fft[i] = output_fft[i] / input_fft[i];





	std::cout.setf(std::ios::fixed);
	fs << "arg\t\tfunct\t\tIFFT\t\tFFT_mkl\t\tFFT_exact\t\tDiff\n";
	for (int i = 0; i < count; i++)
	{
		//nodes - аргумент функции
		//in - значение функции
		//out2 - обраное преобразование Фурье mkl
		//out - преобразование Фурье mkl
		//out1 - преобразование Фурье аналитически

		//fs << nodes[i] << ":\t" << in[i] << ";\t" << out2[i] / count << ":\t" << out[i] / count << ";\t" << out1[i] << ";\t" << "\tDiff: " << abs(out[i] / count - out1[i]) << "\n";
	}

	fs.close();

	DftiFreeDescriptor(&inputF);
	DftiFreeDescriptor(&outputF);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiFreeDescriptor: " << error_message << "\t " << status << "\n";
}



