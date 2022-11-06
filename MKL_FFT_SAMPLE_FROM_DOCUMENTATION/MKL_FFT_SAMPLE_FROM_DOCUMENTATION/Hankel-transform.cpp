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

struct Expression {
	Expression(std::string token) : token(token) {}
	Expression(std::string token, Expression a) : token(token), args{ a } {}
	Expression(std::string token, Expression a, Expression b) : token(token), args{ a, b } {}

	std::string token;
	std::vector<Expression> args;
};

class Parser {
public:
	explicit Parser(const char* input) : input(input) {}
	Expression parse();
private:
	std::string parse_token();
	Expression parse_simple_expression();
	Expression parse_binary_expression(int min_priority);

	const char* input;
};

std::string Parser::parse_token() {
	while (std::isspace(*input)) ++input;

	if (std::isdigit(*input)) {
		std::string number;
		while (std::isdigit(*input) || *input == '.') number.push_back(*input++);
		return number;
	}

	static const std::string tokens[] =
	{ "+", "-", "**", "*", "/", "mod", "abs", "sin", "cos", "(", ")", "ln", "exp" };
	for (auto& t : tokens) {
		if (std::strncmp(input, t.c_str(), t.size()) == 0) {
			input += t.size();
			return t;
		}
	}

	return "";
}

Expression Parser::parse_simple_expression() {
	auto token = parse_token();
	if (token.empty()) throw std::runtime_error("Invalid input");

	if (token == "(") {
		auto result = parse();
		if (parse_token() != ")") throw std::runtime_error("Expected ')'");
		return result;
	}

	if (std::isdigit(token[0]))
		return Expression(token);

	return Expression(token, parse_simple_expression());
}

int get_priority(const std::string& binary_op) {
	if (binary_op == "+") return 1;
	if (binary_op == "-") return 1;
	if (binary_op == "*") return 2;
	if (binary_op == "/") return 2;
	if (binary_op == "mod") return 2;
	if (binary_op == "**") return 3;
	return 0;
}

Expression Parser::parse_binary_expression(int min_priority) {
	auto left_expr = parse_simple_expression();

	for (;;) {
		auto op = parse_token();
		auto priority = get_priority(op);
		if (priority <= min_priority) {
			input -= op.size();
			return left_expr;
		}

		auto right_expr = parse_binary_expression(priority);
		left_expr = Expression(op, left_expr, right_expr);
	}
}

Expression Parser::parse() {
	return parse_binary_expression(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double eval(const Expression& e) {
	switch (e.args.size()) {
	case 2: {
		auto a = eval(e.args[0]);
		auto b = eval(e.args[1]);
		if (e.token == "+") return a + b;
		if (e.token == "-") return a - b;
		if (e.token == "*") return a * b;
		if (e.token == "/") return a / b;
		if (e.token == "**") return pow(a, b);
		if (e.token == "mod") return (int)a % (int)b;
		throw std::runtime_error("Unknown binary operator");
	}

	case 1: {
		auto a = eval(e.args[0]);
		if (e.token == "+") return +a;
		if (e.token == "-") return -a;
		if (e.token == "abs") return abs(a);
		if (e.token == "sin") return sin(a);
		if (e.token == "cos") return cos(a);
		if (e.token == "ln") return log(a);
		if (e.token == "exp") return exp(a);
		throw std::runtime_error("Unknown unary operator");
	}

	case 0:
		return strtod(e.token.c_str(), nullptr);
	}

	throw std::runtime_error("Unknown expression type");
}

std::string StringReplacer(const std::string& inputStr, const std::string& src, const std::string& dst)
{
	std::string result(inputStr);

	size_t pos = result.find(src);
	while (pos != std::string::npos) {
		result.replace(pos, src.size(), dst);
		pos = result.find(src, pos);
	}

	return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef double (*Func) (double x);
const double beta = -0.01;

double expBeta(double x)
{

	return exp(-beta * abs(x));
}

double FFT_expBeta(double w)
{
	return 2 * beta / (beta * beta + w * w);
}


int main() {
	int err = 1;
	//Ввод шага дискретизации
	const double delta = 0.1;

	//Ввод границ интегрирования
	double N = 10;

	//Получение массива значений функции f
	const int count = N / delta;
	double* nodes = new double[count];
	for (int i = 0; i < count; i++)
	{
		nodes[i] = N - delta * i;
	}

	std::string s;
	err = 1;

	int ngrid[] = { count };
	const int howmany = 1;
	const size_t distanse = count;
	double* in = (double*)mkl_malloc(count * sizeof(double), 64);
	double* out = (double*)mkl_malloc(count * sizeof(double), 64);

	for (int i = 0; i < count; i++)
	{
		in[i] = expBeta(nodes[i]);
	}

	DFTI_DESCRIPTOR_HANDLE mkl_plan;

	MKL_LONG status = DftiCreateDescriptor(&mkl_plan, DFTI_DOUBLE, DFTI_REAL,
		1, (MKL_LONG)count);
	std::string error_message;
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCreateDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiSetValue(mkl_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiSetValue: " << error_message << "\t " << status << "\n";

	status = DftiCommitDescriptor(mkl_plan);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiCommitDescriptor: " << error_message << "\t " << status << "\n";

	status = DftiComputeForward(mkl_plan, in, out);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeForward: " << error_message << "\t " << status << "\n";


	double* out1 = (double*)mkl_malloc(count * sizeof(double), 64);
	
	for (int i = 0; i < count; i++)
	{
		out1[i] = FFT_expBeta(nodes[i]);
	}

	double* out2 = (double*)mkl_malloc(count * sizeof(double), 64);
	status = DftiComputeBackward(mkl_plan, out, out2);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiComputeBackward: " << error_message << "\t " << status << "\n";
	std::cout.setf(std::ios::fixed);
	std::cout << "arg\t\tfunc\t\tFFT_mkl\t\tFFT\t\tIFFT\t\tDiff\n";
	
	for (int i = 0; i < count; i++)
	{
		//nodes - аргумент функции
		//in - значение функции
		//out - преобразование Фурье mkl
		//out1 - преобразование Фурье аналитически
		//out2 - обраное преобразование Фурье mkl
		std::cout << nodes[i] << ":\t" << in[i] << ";\t" << out[i] << ";\t" << out1[i] << ";\t" << out2[i] << "\tDiff: " << abs(out[i] - out1[i]) << "\n";
	}

	DftiFreeDescriptor(&mkl_plan);
	error_message = DftiErrorMessage(status);
	std::cout << "DftiFreeDescriptor: " << error_message << "\t " << status << "\n";
}


