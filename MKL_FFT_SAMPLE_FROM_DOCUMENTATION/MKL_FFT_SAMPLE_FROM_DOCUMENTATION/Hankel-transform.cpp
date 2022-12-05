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
#include <ctime>


typedef double (*Func) (double x);
const double beta = 0.01;
const double a = 0.000001;
const double b = 0.000001;
const double z = 0.01;
double PI = 3.14159265358979323846;
std::ofstream  fs("test.txt");
std::ofstream  err_mkl("err_mkl.txt");
int err;

double expBeta(double x)
{
	return exp(-beta * abs(x));
}

double input_func_j0(double x)
{
	return x * exp(-a * pow(x, 2));
}

double output_func_j0(double x)
{
	return exp(-pow(x, 2) / (4 * a)) / (2 * a);
}

double input_func_j1(double x)
{
	return pow(x, 2) * exp(-a * pow(x, 2));
}

double output_func_j1(double x)
{
	return b * exp(-pow(x, 2) / (4 * a)) / (2 * a); //что такое b???
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

double sinc(double x)
{
	if (x != 0) return sin(PI * x) / (PI * x);
	else return 1;
}

/* Compute (K*L)%M accurately */
static float moda(int K, int L, int M)
{
	return (float)(((long long)K * L) % M);
}

/* Initialize array x(N) to produce unit peak at y(H) */
static void init_c(MKL_Complex16* x, int N, int H)
{
	float TWOPI = 6.2831853071795864769f, phase;
	int n;

	for (n = 0; n < N / 2 + 1; n++)
	{
		phase = moda(n, H, N) / N;
		x[n].real = cosf(TWOPI * phase) / N;
		x[n].imag = -sinf(TWOPI * phase) / N;
	}
}

double test_input(double x)
{
	return exp(-x * z);
}

double test_output(double x)
{
	return 1 / sqrt(pow(x, 2) + pow(z, 2));
}

int HankelTransform(double delta, double L, double (*test_input)(double), double (*test_output)(double), int mode)
{
	const int count = 2 * L / delta;
	double* nodes = new double[count];
	for (int i = 0; i < count; i++)
	{
		nodes[i] = (double)delta * (i + 1);
	}

	std::string s;
	err = 1;

	int ngrid[] = { count };
	//const int howmany = 1;
	const size_t distanse = count;

	//FFT для входной функции
	double* input_array = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* input_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);

	for (int i = 0; i < count; i++)
		input_array[i] = mode == 0 ? input_func_j0(exp(nodes[i])) : input_func_j1(exp(nodes[i]));

	DFTI_DESCRIPTOR_HANDLE inputF;

	MKL_LONG status = DftiCreateDescriptor(&inputF, DFTI_DOUBLE, DFTI_REAL,
		1, (MKL_LONG)count);
	std::string error_message;

	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(inputF, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(inputF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(inputF);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(inputF, input_array, input_fft);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	init_c(input_fft, count, -1);


	//FFT для выходной функции
	double* output_array = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* output_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);

	for (int i = 0; i < count; i++)
		output_array[i] = exp(nodes[i]) * (mode == 0 ? output_func_j0(exp(nodes[i])) : output_func_j1(exp(nodes[i])));

	DFTI_DESCRIPTOR_HANDLE outputF;
	status = DftiCreateDescriptor(&outputF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(outputF, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(outputF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(outputF);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(outputF, output_array, output_fft);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	init_c(output_fft, count, -1);

	//Получаем преобразование Фурье Функции Бесселя по теореме о свертке
	MKL_Complex16* bessel_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);
	for (int i = 0; i < (count / 2 + 1); i++)
	{
		bessel_fft[i].real = (output_fft[i].real * input_fft[i].real + output_fft[i].imag * input_fft[i].imag) / (pow(input_fft[i].real, 2) + pow(input_fft[i].imag, 2));
		bessel_fft[i].imag = (input_fft[i].real * output_fft[i].imag - output_fft[i].real * input_fft[i].imag) / (pow(input_fft[i].real, 2) + pow(input_fft[i].imag, 2));
	}

	//FFT для sinc(x)
	double* _sinc = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* sinc_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);

	for (int i = 0; i < count; i++)
		_sinc[i] = sinc(nodes[i]);

	DFTI_DESCRIPTOR_HANDLE sinc_task;
	status = DftiCreateDescriptor(&sinc_task, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(sinc_task, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(sinc_task, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(sinc_task);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(sinc_task, _sinc, sinc_fft);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	init_c(sinc_fft, count, -1);

	//Вычисление фильтра
	double* W = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* W_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);

	for (int i = 0; i < (count / 2 + 1); i++)
	{
		W_fft[i].real = sinc_fft[i].real * bessel_fft[i].real - sinc_fft[i].imag * bessel_fft[i].imag;
		W_fft[i].imag = sinc_fft[i].real * bessel_fft[i].imag + sinc_fft[i].imag * bessel_fft[i].real;
	}

	DFTI_DESCRIPTOR_HANDLE filter_task;
	status = DftiCreateDescriptor(&filter_task, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(filter_task, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(filter_task, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(filter_task);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeBackward(filter_task, W_fft, W);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiComputeForward: " << error_message << "\t " << status << "\n";


	std::cout.setf(std::ios::fixed);
	fs << "arg\t\tfilter\t\t\tW\n";

	fs << "--------------------------------------------------\n";

	double* test_output_array_mkl = (double*)mkl_malloc(count * sizeof(double), 64);
	double* test_output_array = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
	{
		test_output_array[i] = test_output(exp(nodes[i]));
		test_output_array_mkl[i] = 0;

		for (int j = 0; j < count; j++)
		{
			test_output_array_mkl[i] += W[j] * test_input(exp(-j * delta - nodes[i])) / exp(nodes[i]);
		}

		fs << nodes[i] << ":\t\t" << abs(test_output_array[i] - test_output_array_mkl[i]) << "\t\t\t" << W[i] << "\n";
	}


	DftiFreeDescriptor(&inputF);
	DftiFreeDescriptor(&outputF);
	error_message = DftiErrorMessage(status);
	err_mkl << "DftiFreeDescriptor: " << error_message << "\t " << status << "\n";

	return 0;
}


int main() {
	unsigned int start_time = clock(); // начальное время

	HankelTransform(0.1, 100, test_input, test_output, 0);

	fs.close();
	err_mkl.close();
	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time; // искомое время
	std::cout << "Program running time: " << (float)search_time / 1000 << " sec\n";
}



