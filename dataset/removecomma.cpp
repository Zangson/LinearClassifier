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

using namespace std;

int main(int argc, char **argv)
{
	FILE *f;
	f = fopen(argv[1], "r");
	char c;
	while(fscanf(f, "%c", &c)!=EOF)
	{
		if(c == ',')
			cout<<" ";
		else
			cout<<c;
	}
	return 0;
}
