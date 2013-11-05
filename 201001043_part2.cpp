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

using namespace std;

vector<vector<int> > train_data, test_data;
vector<vector<double> > train_set, test_set;
vector<vector<double> > test;
vector<int> train_label, test_label;
vector<int> misclassified;
map<int, int> misclass;
vector<vector<double> > classifiers;

void matrix_mean()
{
	for(int i=0; i<train_data.size(); i++)
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
						mean += (double)train_data[i][l*28+m];
					}
				}
				temp.push_back(mean);
			}
		}
		temp.push_back(1.0);
		train_set.push_back(temp);
	}

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

vector<vector<double> > negatify(vector<vector<double> > train, int class_no, vector<int> label)
{
	vector<vector<double> > train2;
	int count = 0;
	for(int i=0; i<train.size(); i++)
	{
		if(label[i] == class_no)
		{
			count++;
			for(int j=0; j<train[i].size(); j++)
				train[i][j] = train[i][j] * (-1.0);
		}
	}
	return train;
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

vector<double> batch_perceptron_training(vector<vector<double> > train)
{
	double eta = 0.2;
	vector<double> a;
	a.clear();

	double theta = 0.5;

	for(int i=0; i<train[0].size(); i++)
	{
		a.push_back(1.0);
	}

	int k=0; 
	int count = 0, miss_count;
	while(count<100000)
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

	return a;
}


vector<double> batch_perceptron_margin_training(vector<vector<double> > train)
{
	double b = 0.4;
	double eta = 0.01;
	vector<double> a;
	a.clear();

	double theta = 0.5;

	for(int i=0; i<train[0].size(); i++)
	{
		a.push_back(1.0);
	}

	int k=0; 
	int count = 0, miss_count;
	while(count<100000)
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

		if(mod(check_stop)<theta)
		{
			break;
		}


	}
	return a;
}

vector<double> batch_relaxation_margin_training(vector<vector<double> > train)
{
	double b = 0.4;
	double eta = 0.01;
	vector<double> a;
	a.clear();

	double theta = 0.5;

	for(int i=0; i<train[0].size(); i++)
	{
		a.push_back(1.0);
	}

	int k=0; 
	int count = 0, miss_count;
	while(count<100000)
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
	return a;
}

vector<double> single_sample_margin_training(vector<vector<double> > train)
{
	double b = 0.1;
	double eta = 0.2;
	vector<double> a;
	a.clear();

	for(int i=0; i<train[0].size(); i++)
		a.push_back(1.0);
	int k=0;
	int count = 0;
	int miss_count;
	while(count<1000000)
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
		if(miss_count <= 10 && count >= 100000)
			break;

		k = (k+1)%train.size();
		if(misclass[k] == 1)
		{
			for(int i=0; i<a.size(); i++)
			{
				a[i] = a[i] + train[k][i];
			}
		}
	}
	return a;
	
}

vector<double> single_sample_training(vector<vector<double> > train)
{

	vector<double> a;
	a.clear();

	for(int i=0; i<train[0].size(); i++)
		a.push_back(1.0);
	
	int k=0;
	int count = 0;
	int miss_count;
	while(count<1000000)
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
		if(miss_count <= 1)
			break;

		k = (k+1)%train.size();
		if(misclass[k] == 1)
		{
			for(int i=0; i<a.size(); i++)
			{
				a[i] = (a[i] + train[k][i]);
			}
		}
	}

	return a;
		
}



void read_data()
{
	ifstream f;
	f.open("dataset/mnistTrainData.txt", ios::in);
	for(int i=0; i<1265; i++)
	{
		vector<int> temp;
		for(int j=0; j<784; j++)
		{
			int x;
			f>>x;
			temp.push_back(x);
		}
		train_data.push_back(temp);
	}

	f.close();
	f.open("dataset/mnistTestData.txt", ios::in);
	for(int i=0; i<1014; i++)
	{
		vector<int> temp;
		for(int j=0; j<784; j++)
		{
			int x;
			f>>x;
			temp.push_back(x);
		}
		test_data.push_back(temp);
	}	
	f.close();

	int label;
	f.open("dataset/mnistTestLabel.txt", ios::in);
	while((f>>label).good())
	{
		test_label.push_back(label);
	}
	f.close();

	f.open("dataset/mnistTrainLabel.txt", ios::in);
	while((f>>label).good())
	{
		train_label.push_back(label);
	}
	f.close();

}

double testing()
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
		if(!flag)
			unclassified++;

		if(max == test_label[n])
			accuracy = accuracy + 1.0;
	}
	return accuracy*100.0/(float)(test_set.size());
				

}

void pairwise_compute(int choice)
{
	vector<vector<double> > train, train2;
	vector<int> label;

	for(int i=0; i<=9; i++)
	{
		for(int j=i+1; j<=9; j++)
		{
			train.clear();
			label.clear();
			for(int k=0; k<train_label.size(); k++)
			{
				if(train_label[k] == i || train_label[k] == j)
				{
					train.push_back(train_set[k]);
					label.push_back(train_label[k]);
				}
			}
			train = negatify(train, i, label);

			vector<double> classify;
			switch(choice)
			{
				case 1: classify = single_sample_training(train);
								break;
				case 2: classify = batch_perceptron_training(train);
								break;
				case 3: classify = single_sample_margin_training(train);
								break;
				case 4: classify = batch_perceptron_margin_training(train);
								break;
				case 5: classify = batch_relaxation_margin_training(train);
								break;
			}
			float accuracy = 0.0;
			classifiers.push_back(classify);
		}				
	}
}

int main(int argc, char **argv)
{
	vector<vector<double> > a;
	read_data();
	matrix_mean();
	int choice = atoi(argv[1]);
	pairwise_compute(choice);
	cout<<testing()<<endl;
	return 0;
}
