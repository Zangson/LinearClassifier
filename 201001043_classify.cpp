#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<cstring>
#include<string>
#include<vector>
#include<map>
#include<stack>
#include<queue>
#include<algorithm>
#include<fstream>
#include<armadillo>

using namespace std;
using namespace arma;

vector<vector<int> > train_data, test_data;
vector<vector<double> > train_set, test_set;
vector<vector<double> > test;
vector<int> train_label, test_label;
vector<int> misclassified;
map<int, int> misclass;
vector<vector<double> > classifiers;

void matrix_mean()
{
	for(int i=0; i<test_data.size(); i++)
	{
		vector<double> temp;
		for(int j=0; j<28; j+=4)
		{
			for(int k=0; k<28; k+=4)
			{
				double mean = 0.0;
				for(int l=j; l<j+4; l++)
				{
					for(int m=k; m<k+4; m++)
					{
						mean += (double)test_data[i][l*28+m];
					}
				}
				temp.push_back(mean);
			}
		}
		temp.push_back(1.0);
		test_set.push_back(temp);
	}
}

double dot_product(vector<double> x, vector<double> y)
{
	double ans = 0.0;
	for(int i=0; i<x.size(); i++)
	{
		ans = ans + x[i]*y[i];
	}
	return ans;
}



void read_data(char filename[])
{
	ifstream f;
	f.open(filename, ios::in);
	int x;
	while((f>>x).good())
	{
		vector<int> temp;
		temp.push_back(x);
		for(int j=0; j<783; j++)
		{
			f>>x;
			temp.push_back(x);
		}
		test_data.push_back(temp);
	}
		
	f.close();

}

void testing()
{
	double accuracy = 0.0;
	int unclassified = 0;
	for(int n=0; n<test_set.size(); n++)
	{
		int count = 0;
		int result[10] = {0};
		count = 0;
		for(int i=0; i<=9; i++)
		{
			for(int j=i+1; j<=9; j++)
			{
				if(dot_product(classifiers[count++], test_set[n]) < 0)
					result[i]++;
				else
					result[j]++;
			}
		}
		int max = 0;
		for(int i=0; i<9; i++)
		{
			if(result[i]>result[max])
				max = i;
		}
		int flag = 1;
		for(int i=0; i<9; i++)
		{
			if(result[i] == result[max] && max!=i)
			{
				flag = 0;
				break;
			}
		}

		cout<<max<<endl;
	}
}


void read_vectors(int choice)
{
	char filename[100];
	switch(choice)
	{
		case 1: strcpy(filename, "single_sample_vectors.txt");
						break;

		case 2: strcpy(filename, "batch_perceptron_vectors.txt");
						break;

		case 3: strcpy(filename, "single_sample_margin_vectors.txt");
						break;

		case 4: strcpy(filename, "batch_perceptron_margin_vectors.txt");
						break;

		case 5: strcpy(filename, "batch_relaxation_margin_vectors.txt");
						break;
	}
	double x;
	ifstream f;
	f.open(filename, ios::in);
	for(int i=0; i<45; i++)
	{
		vector<double> temp;
		for(int j=0; j<50; j++)
		{
			double x;
			f>>x;
			temp.push_back(x);
		}
		classifiers.push_back(temp);
	}
}

int main(int argc, char **argv)
{
	vector<vector<double> > a;
	if(argc < 3)
	{
		cout<<"Invalid usage\n";
		exit(1);
	}
	read_data(argv[2]);
	matrix_mean();
	int choice = atoi(argv[1]);
	if(choice<1 || choice > 5)
	{
		cout<<"Invalid Usage\n";
		exit(1);
	}
	read_vectors(choice);
	testing();
	return 0;
}
