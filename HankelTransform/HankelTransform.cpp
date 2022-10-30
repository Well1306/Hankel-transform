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

int main() {
	int err = 1;
	//Ввод шага дискретизации
	double delta;
	std::string _delta;
	err = 1;
	while (err)
	{
		std::cout << "Enter the sampling step: ";
		std::cin >> _delta;
		try
		{
			delta = stod(_delta);
			err = 0;
		}
		catch (std::exception& e)
		{
			std::cout << "Error: Need a number" << "\n";
			continue;
		}
	}

	//Ввод границ интегрирования
	int N;
	std::string _N;
	err = 1;
	while (err)
	{
		std::cout << "Enter integration bounds: ";
		std::cin >> _N;
		try
		{
			N = stoi(_N);
			err = 0;
		}
		catch (std::exception& e)
		{
			std::cout << "Error: Need a number" << "\n";
			continue;
		}
	}

	//Получение массива значений функции f
	const int count = N / delta + 1;
	double* nodes = new double[count];
	double* sum_f = new double[count];
	for (int i = 0; i < count; i++)
	{
		nodes[i] = N - delta * i;
	}

	std::string s;
	err = 1;
	while (err)
	{
		std::cout << "Enter integrand function f(t): ";
		std::cin >> s;
		std::string _s = StringReplacer(s, "t", "0");
		try
		{
			Parser p(_s.c_str());
			auto result = eval(p.parse());
			err = 0;
		}
		catch (std::exception& e)
		{
			std::cout << "Error: function entry error" << "\n";
			continue;
		}
	}

	for (int i = 0; i < count; i++)
	{
		std::string _s = StringReplacer(s, "t", std::to_string(nodes[i]));
		Parser p(_s.c_str());
		auto result = eval(p.parse());
		sum_f[i] = result;
	}
	for (int i = count - 1; i >= 0; i--)
	{
		//std::cout << nodes[i] << ": " << sum_f[i] << "\n";
	}

	int ngrid[] = { count };
	const int howmany = 1;
	const size_t distanse = 11;
	_MKL_Complex16 in[howmany * distanse];
	_MKL_Complex16 out[howmany * distanse];

	for (int i = 0; i < 11; i++)
	{
		in[i].real = sum_f[i];
		in[i].imag = 0;
	}

	DFTI_DESCRIPTOR_HANDLE mkl_plan;

	MKL_LONG status = DftiCreateDescriptor(&mkl_plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, ngrid);
	status = DftiSetValue(mkl_plan, DFTI_NUMBER_OF_TRANSFORMS, howmany);
	status = DftiSetValue(mkl_plan, DFTI_INPUT_DISTANCE, distanse);
	DftiCommitDescriptor(mkl_plan);

	DftiComputeForward(mkl_plan, in, out);

	for (int i = 0; i < count; i++)
	{
		std::cout << nodes[i] << ": " << in[i].real << "; " << out[i].real << "\n";
	}

	DftiFreeDescriptor(&mkl_plan);
	//std::string s;
	//std::cin >> s;
	//s = StringReplacer(s, "x", "1");
	//Parser p(s.c_str());
	//auto result = eval(p.parse());
	//std::cout << result;
}
