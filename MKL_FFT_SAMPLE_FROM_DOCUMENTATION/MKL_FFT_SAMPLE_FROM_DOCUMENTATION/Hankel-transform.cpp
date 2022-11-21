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
const double a = 0.000001;
double PI = 3.14159265358979323846;

double expBeta(double x)
{
	return exp(-beta * abs(x));
}

double input_func(double x)
{
	return x * exp(-a * pow(x, 2));
}

double output_func(double x)
{
	return exp(-pow(x, 2) / (4 * a)) / (2 * a);
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
	return exp(-x * 0.001);
}

double test_output(double x)
{
	return 1 / sqrt(pow(x, 2) + pow(0.001, 2));
}

int main() {
	/* Arbitrary harmonic used to verify FFT */
	int H = -1;

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
	const double delta = 0.01;

	//Ввод границ интегрирования
	//double N = 10;
	double L = 100;

	//Получение массива значений функции f
	const int count = 2 * L / delta;
	double* nodes = new double[count];
	for (int i = 0; i < count; i++)
	{
		nodes[i] = (double)-L + delta * i;
	}

	std::string s;
	err = 1;

	int ngrid[] = { count };
	//const int howmany = 1;
	const size_t distanse = count;

	//FFT для входной функции
	double* input_array = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* input_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);
	double* input_ifft = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
		input_array[i] = input_func(nodes[i]);

	DFTI_DESCRIPTOR_HANDLE inputF;

	MKL_LONG status = DftiCreateDescriptor(&inputF, DFTI_DOUBLE, DFTI_REAL,
		1, (MKL_LONG)count);
	std::string error_message;

	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(inputF, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(inputF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(inputF);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(inputF, input_array, input_fft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	init_c(input_fft, count, H);

	status = DftiComputeBackward(inputF, input_fft, input_ifft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeBackward: " << error_message << "\t " << status << "\n";


	//FFT для выходной функции
	double* output_array = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* output_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);
	double* output_ifft = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
		output_array[i] = output_func(nodes[i]);

	DFTI_DESCRIPTOR_HANDLE outputF;
	status = DftiCreateDescriptor(&outputF, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(outputF, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(outputF, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(outputF);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(outputF, output_array, output_fft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	init_c(output_fft, count, H);

	status = DftiComputeBackward(outputF, output_fft, output_ifft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeBackward: " << error_message << "\t " << status << "\n";


	//Получаем преобразование Фурье Функции Бесселя по теореме о свертке
	MKL_Complex16* bessel_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);
	for (int i = 0; i < (count / 2 + 1); i++)
	{
		bessel_fft[i].real = output_fft[i].real / input_fft[i].real;
		bessel_fft[i].imag = output_fft[i].imag / input_fft[i].imag;
	}

	//FFT для sinc(x)
	double* _sinc = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* sinc_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);

	for (int i = 0; i < count; i++)
		_sinc[i] = sinc(nodes[i]);

	DFTI_DESCRIPTOR_HANDLE sinc_task;
	status = DftiCreateDescriptor(&sinc_task, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(sinc_task, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(sinc_task, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(sinc_task);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(sinc_task, _sinc, sinc_fft);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";

	init_c(sinc_fft, count, H);

	//Вычисление фильтра
	double* W = (double*)mkl_malloc(count * sizeof(double), 64);
	MKL_Complex16* W_fft = (MKL_Complex16*)mkl_malloc((count / 2 + 1) * sizeof(MKL_Complex16), 64);

	for (int i = 0; i < (count / 2 + 1); i++)
	{
		W_fft[i].real = sinc_fft[i].real * bessel_fft[i].real;
		W_fft[i].imag = sinc_fft[i].imag * bessel_fft[i].imag;
	}
	
	DFTI_DESCRIPTOR_HANDLE filter_task;
	status = DftiCreateDescriptor(&filter_task, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)count);

	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(filter_task, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(filter_task, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(filter_task);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeBackward(filter_task, W_fft, W);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";


	std::cout.setf(std::ios::fixed);
	fs << "arg\t\tfilter\n";
	for (int i = 0; i < count; i++)
	{
		//nodes - аргумент функции
		//in - значение функции
		//out2 - обраное преобразование Фурье mkl
		//out - преобразование Фурье mkl
		//out1 - преобразование Фурье аналитически

		//fs << nodes[i] << ":\t" << W[i] << "\n";
	}

	fs << "\n--------------------------------------------------\n";

	double* test_output_array_mkl = (double*)mkl_malloc(count * sizeof(double), 64);
	double* test_output_array = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
	{
		test_output_array[i] = test_output(nodes[i]);
		test_output_array_mkl[i] = 0;

		for (int j = 0; j < count; j++)
		{
			test_output_array_mkl[i] += W[j] * test_input(exp(-j * delta - log(nodes[i]))) / nodes[i];
		}

		fs << nodes[i] << ":\t" << abs(test_output_array[i] - test_output_array_mkl[i]) << "\n";
	}




	fs.close();

	DftiFreeDescriptor(&inputF);
	DftiFreeDescriptor(&outputF);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiFreeDescriptor: " << error_message << "\t " << status << "\n";
}



