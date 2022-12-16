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
#include <complex>


typedef double (*Func) (double x);
const double beta = 0.01;
const double a = 0.01;
const double b = 0.1;
const double z = 100;
double PI = 3.14159265358979323846;
std::ofstream  fs("test4.txt");
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
	return b * exp(-pow(x, 2) / (4 * a)) / pow((2 * a), 2); //что такое b???
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

double test_output1(double x)
{
	return (1 - z / sqrt(pow(x, 2) + pow(z, 2))) / x;
}

//helper function for fft and ifft:
DFTI_DESCRIPTOR* create_descriptor(MKL_LONG length) {
	DFTI_DESCRIPTOR* handle = nullptr;
	// using DFTI_DOUBLE for double precision
	// using DFTI_REAL for using the real version
	bool valid = (DFTI_NO_ERROR == DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 1, length)) &&
		// the result should not be inplace:
		(DFTI_NO_ERROR == DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)) &&
		// make clear that the result should be a vector of complex:
		(DFTI_NO_ERROR == DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
	// chosen normalization is fft(constant)[0] = constant:
	(DFTI_NO_ERROR == DftiSetValue(handle, DFTI_FORWARD_SCALE, 1. / length)) &&
		(DFTI_NO_ERROR == DftiCommitDescriptor(handle));
	if (!valid) {
		DftiFreeDescriptor(&handle);
		return nullptr; //nullptr means error
	}
	return handle;
}

std::vector<std::complex<double>> real_fft(std::vector<double>& in) {
	size_t out_size = in.size() / 2 + 1; //so many complex numbers needed
	std::vector<std::complex<double>> result(out_size);
	DFTI_DESCRIPTOR* handle = create_descriptor(static_cast<MKL_LONG>(in.size()));
	bool valid = handle &&
		(DFTI_NO_ERROR == DftiComputeForward(handle, in.data(), result.data()));
	if (handle) {
		valid &= (DFTI_NO_ERROR == DftiFreeDescriptor(&handle));
	}
	if (!valid) {
		result.clear(); //empty vector -> error
	}
	return result;
}

std::vector<double> real_ifft(std::vector<std::complex<double>>& in, size_t original_size) {
	size_t expected_size = original_size / 2 + 1;
	if (expected_size != in.size()) {
		return {};// empty vector -> error
	}
	std::vector<double> result(original_size);
	DFTI_DESCRIPTOR* handle = create_descriptor(static_cast<MKL_LONG>(original_size));
	bool valid = handle &&
		(DFTI_NO_ERROR == DftiComputeBackward(handle, in.data(), result.data()));
	if (handle) {
		valid &= (DFTI_NO_ERROR == DftiFreeDescriptor(&handle));
	}
	if (!valid) {
		result.clear(); //empty vector -> error
	}
	return result;
}

std::vector<double> MakeFilter(double delta, double L, int mode)
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
	std::vector<double> input_array;
	input_array.resize(count);
	std::vector<std::complex<double>> input_fft;
	input_fft.resize(count / 2 + 1);

	for (int i = 0; i < count; i++)
	{
		input_array[i] = mode == 0 ? input_func_j0(nodes[i]) : input_func_j1(nodes[i]);
	}

	input_fft = real_fft(input_array);


	MKL_LONG status;
	std::string error_message;

	//FFT для выходной функции
	std::vector<double> output_array;
	output_array.resize(count);
	std::vector<std::complex<double>> output_fft;
	output_fft.resize(count / 2 + 1);

	for (int i = 0; i < count; i++)
	{
		output_array[i] = nodes[i] * (mode == 0 ? output_func_j0(nodes[i]) : output_func_j1(nodes[i]));
	}
	output_fft = real_fft(output_array);


	//Получаем преобразование Фурье Функции Бесселя по теореме о свертке
	std::vector<std::complex<double>> bessel_fft;
	bessel_fft.resize(count / 2 + 1);
	for (int i = 0; i < (count / 2 + 1); i++)
	{
		bessel_fft[i] = output_fft[i] / input_fft[i];
	}

	//FFT для sinc(x)
	std::vector<double>  _sinc;
	_sinc.resize(count);
	std::vector<std::complex<double>> sinc_fft;
	sinc_fft.resize(count / 2 + 1);

	for (int i = 0; i < count; i++)
		_sinc[i] = sinc(nodes[i]);

	sinc_fft = real_fft(_sinc);


	//Вычисление фильтра
	std::vector<double> W;
	W.resize(count);
	std::vector<std::complex<double>> W_fft;
	W_fft.resize(count / 2 + 1);

	for (int i = 0; i < (count / 2 + 1); i++)
	{
		W_fft[i] = sinc_fft[i] * bessel_fft[i];
	}

	W = real_ifft(W_fft, count);

	
	return W;
}

int HankelTransform(std::vector<double> W, double delta, double L, double (*test_input)(double))
{
	const int count = 2 * L / delta;
	double* nodes = new double[count];
	for (int i = 0; i < count; i++)
	{
		nodes[i] = (double)delta * (i + 1);
	}

	std::vector<double> test_output_array_mkl;
	test_output_array_mkl.resize(count);

	std::cout.setf(std::ios::fixed);
	fs << "arg\t\tfilter\t\t\tW\n";

	fs << "--------------------------------------------------\n";
	for (int i = 0; i < count; i++)
	{
		test_output_array_mkl[i] = 0;

		for (int j = 0; j < count; j++)
		{
			test_output_array_mkl[i] += W[j] * test_input(exp(-j * delta - log(nodes[i]))) / nodes[i];
		}
		//fs << test_output_array_mkl[i] << "\n";
	}
	return 0;
}


int main() {
	unsigned int start_time = clock(); // начальное время

	double delta = 0.01;
	double L = 1;
	int count = 2 * L / delta;
	std::vector<double> W;
	W.resize(count);
	W = MakeFilter(delta, L, 0);
	
	std::cout << "z = " << z << "\n";

	//for (int j = 0; j < count; j++)
	//{
	//	fs << delta * (j + 1) << "\n"; //nodes[i]
	//}
	
	fs << "---------------------------------\n";

	HankelTransform(W, delta, L, test_input);

	fs.close();
	err_mkl.close();
	unsigned int end_time = clock(); // конечное время
	unsigned int search_time = end_time - start_time; // искомое время
	std::cout << "Program running time: " << (float)search_time / 1000 << " sec\n";
}



