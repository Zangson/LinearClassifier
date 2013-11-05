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

vector<vector<double> > train, test;
vector<int> train_label, test_label;
vector<int> misclassified;
map<int, int> misclass;
vector<double> a;

void negatify()
{
	for(int i=0; i<train.size(); i++)
	{
		if(train_label[i] == 0)
		{
			train[i][0] = train[i][0]*(-1.0);
			train[i][1] = train[i][1]*(-1.0);
			train[i][2] = train[i][2]*(-1.0);
		}
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

int less_than(vector<double> x, vector<double> y)
{
	int i, dim = x.size();
	for(i=0; i<dim; i++)
	{
		if(x[i] > y[i])
			break;
	}

	if(i<dim)
		return 0;

	else
		return 1;
}

double mod(vector<double> x)
{
	double ans = 0;
	for(int i=0; i<x.size(); i++)
		ans += x[i]*x[i];
	return sqrt(ans);
}

void batch_perceptron_training(double eta = 1.0)
{
	a.clear();

	double theta = 0.5;

	for(int i=0; i<train[0].size(); i++)
	{
		a.push_back(1.0);
	}

	int k=0; 
	int count = 0, miss_count;
	while(count<1000)
	{
		count++;
		k++;
		misclassified.clear();
		miss_count = 0;
		vector<double> check_stop;
		for(int i=0; i<train[0].size(); i++)
			check_stop.push_back(0.0);
		for(int i=0; i<train.size(); i++)
		{
			if(dot_product(a, train[i]) < 0)
			{
				for(int j=0; j<train[i].size(); j++)
					check_stop[j] = check_stop[j] + train[i][j];
				miss_count++;
			}
		}
		if(miss_count <= 0)
		{
			break;
		}
		
		for(int i=0; i<check_stop.size(); i++)
		{
			check_stop[i] *= eta;
			a[i] = a[i] + check_stop[i];
		}

		if(mod(check_stop)<theta)
		{
			break;
		}
	}

}

void batch_perceptron_margin_training(double b=0.4, double eta=0.1)
{
	a.clear();

	double theta = 0.5;

	for(int i=0; i<train[0].size(); i++)
	{
		a.push_back(1.0);
	}

	int k=0; 
	int count = 0, miss_count;
	while(count<1000)
	{
		k = (k+1);
		count++;
		misclassified.clear();
		miss_count = 0;
		vector<double> check_stop;
		for(int i=0; i<train[0].size(); i++)
			check_stop.push_back(0.0);
		for(int i=0; i<train.size(); i++)
		{
			if(dot_product(a, train[i]) < b)
			{
				for(int j=0; j<train[i].size(); j++)
					check_stop[j] = check_stop[j] + train[i][j];
				miss_count++;
			}
		}
		if(miss_count <= 0)
		{
			break;
		}
		
		for(int i=0; i<check_stop.size(); i++)
		{
			a[i] = a[i] + eta*check_stop[i];
		}

	}

}

void batch_relaxation_margin_training(double b=0.4, double eta=0.1)
{
	a.clear();

	double theta = 0.5;

	for(int i=0; i<train[0].size(); i++)
	{
		a.push_back(1.0);
	}

	int k=0; 
	int count = 0, miss_count;
	while(count<1000)
	{
		k = (k+1);
		count++;
		misclassified.clear();
		miss_count = 0;
		vector<double> check_stop;
		for(int i=0; i<train[0].size(); i++)
			check_stop.push_back(0.0);
		for(int i=0; i<train.size(); i++)
		{
			if(dot_product(a, train[i]) < b)
			{
				for(int j=0; j<train[i].size(); j++)
					check_stop[j] = check_stop[j] + ((b-dot_product(a, train[i])/(mod(train[i])*mod(train[i])))*train[i][j]);
				miss_count++;
			}
		}
		if(miss_count <= 0)
		{
			break;
		}
		
		for(int i=0; i<check_stop.size(); i++)
		{
			a[i] = a[i] + eta*check_stop[i];
		}

	}

}

void single_sample_margin_training(double b=0.2, double eta=0.2)
{
	a.clear();

	for(int i=0; i<train[0].size(); i++)
		a.push_back(1.0);
	int k=0;
	int count = 0;
	int miss_count;
	while(count<1000)
	{
		count++;
		misclass.clear();
		miss_count = 0;
		for(int i=0; i<train.size(); i++)
		{
			if(dot_product(a, train[i]) < b)
			{
				miss_count++;
				misclass[i] = 1;
			}
		}
		if(miss_count <= 2)
			break;

		k = (k+1)%train.size();
		if(misclass[k] == 1)
		{
			for(int i=0; i<a.size(); i++)
			{
				a[i] = a[i] + eta*train[k][i];
			}
		}
	}
	
}

void single_sample_training()
{

	a.clear();

	for(int i=0; i<train[0].size(); i++)
		a.push_back(1.0);
	int k=0;
	int count = 0;
	int miss_count;
	while(count<1000)
	{
		count++;
		misclass.clear();
		miss_count = 0;
		for(int i=0; i<train.size(); i++)
		{
			if(dot_product(a, train[i]) < 0.0)
			{
				miss_count++;
				misclass[i] = 1;
			}
		}
		if(miss_count == 0)
			break;

		k = (k+1)%train.size();
		if(misclass[k] == 1)
		{
			for(int i=0; i<a.size(); i++)
			{
				a[i] = (a[i] + 0.5*train[k][i]);
			}
		}
	}
		
}


double testing()
{
	double accuracy = 0.0;
	for(int i=0; i<test.size(); i++) 
	{
		double dp = dot_product(a, test[i]);
		int result;
		if(dp > 0)
			result = 1;
		else
			result = 0;
		if(result == test_label[i])
			accuracy = accuracy + 1.0;
	}
	return accuracy*100.0/(double)test.size();
}

void read_data()
{
	ifstream f;
	f.open("dataset/trainData.txt", ios::in);
	double x,y;
	while((f>>x>>y).good())
	{
		vector<double> temp;
		temp.push_back(x);
		temp.push_back(y);
		temp.push_back(1.0);
		train.push_back(temp);
	}
	f.close();
	f.open("dataset/testData.txt", ios::in);
	while((f>>x>>y).good())
	{
		vector<double> temp;
		temp.push_back(x);
		temp.push_back(y);
		temp.push_back(1.0);
		test.push_back(temp);
	}
	f.close();

	int label;
	f.open("dataset/testLabel.txt", ios::in);
	while((f>>label).good())
	{
		test_label.push_back(label);
	}
	f.close();

	f.open("dataset/trainLabel.txt", ios::in);
	while((f>>label).good())
	{
		train_label.push_back(label);
	}
	f.close();

}

int main(int argc, char **argv)
{
	if(argc<2)
	{
		cout<<"Invalid Usage"<<endl;
		exit(1);
	}
	read_data();
	negatify();
	int choice = atoi(argv[1]);
	switch(choice)
	{

		case 1: single_sample_training();
						break;

		case 2: batch_perceptron_training();
						break;

		case 3: single_sample_margin_training();
						break;
	
		case 4: batch_perceptron_margin_training();
						break;

		case 5: batch_relaxation_margin_training();
						break;

		default: cout<<"Invalid choice\n";
						 exit(1);
						 break;
		 
	}

	cout<<"Accuracy= "<<testing()<<endl;
	return 0;
}
