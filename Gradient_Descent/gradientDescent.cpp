/*Project designed to compute gradient descent (Machine Learning) on Intel HD 4000 GPU */

#include<iostream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cstdio>
#include<fstream>
#include<CL/cl.hpp>
#include<sstream>
#include<string>

#define SIZE 96

using namespace std;
int main(){
//	vector<float> x;
//	vector<float> y;
	float x[SIZE],y[SIZE];
	float temp[SIZE/4];
	float temp1=0,temp2=0;
	
	FILE *fp,*fp2;
	fp=fopen("ex1data1.txt", "r");
	fp2=fopen("data.txt","w");
	for (int i=0;!feof(fp);i++){
		fscanf(fp,"%f,%f",&temp1,&temp2);
		fprintf(fp2,"%f\t%f\n",temp1,temp2);
//		x.push_back(temp1);
//		y.push_back(temp2);
		x[i]=temp1;
		y[i]=temp2;
		
	}
	fclose(fp);
	fclose(fp2);
	cout<<"\nFile read\n";
/*	
	FILE *pipe=popen("gnuplot -persistent","w");
	if (pipe == NULL){
		cout<<"\nFailed";
	}
//	fprintf(pipe,"set xrange[0:24]\n");
//	fprintf(pipe,"set yrange[-5:27]\n");
	fprintf(pipe,"plot 'data.txt' using 1:2\n");
	fclose(pipe);	
	getchar();
*/	

	////////////Finding Platforms/////////////////////
	vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if(all_platforms.size()==0){
		cout<<"No platform found!!\n";
		exit(1);
	}
	cout<<"Available Platforms:\n";
	for(cl::Platform a: all_platforms){
		cout<<a.getInfo<CL_PLATFORM_NAME>()<<"\n";;
	}
	cl::Platform default_platform=all_platforms.at(0);
	cout<<"Using Platform:"<<default_platform.getInfo<CL_PLATFORM_VERSION>()<<"\n";
	
	///////////Finding Devices////////////////////////
	vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL,&all_devices);
	if(all_devices.size()==0){
		cout<<"No Devices found!!\n";
		exit(1);
	}
	
	cout<<"Available Devices:\n";
	for(cl::Device a: all_devices){
		cout<<a.getInfo<CL_DEVICE_NAME>()<<"\n";
	}
	cl::Device default_device=all_devices.at(0);
//	cout<<"Using Device:"<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
	cout<<"Using Device:"<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
	vector<size_t> n=default_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
	
	cout<<"Available size: (";
	for(size_t k: n){
		cout<<k<<",";
	}
	cout<<")"<<"\n";
	cl::Context context(default_device);//Can use arg (all_devices);
	
	/////////////Read KERNEL FILE////////////////////////
	cl::Program::Sources source;
	string kernel_code;
	ifstream ifs("kernel.cl");
	kernel_code.assign((istreambuf_iterator<char>(ifs)),(std::istreambuf_iterator<char>()));
	source.push_back({kernel_code.c_str(), kernel_code.length()});
	
	cl::Program program(context, source);
	if(program.build({default_device})<0){
		cout<<"Error building:"<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
		exit(1);
	}
	cout<<"\nAllocating Buffers";
	/////////////Allocating Buffers on Device////////////////////////
	
	cl::Buffer buffer_X(context,CL_MEM_READ_ONLY, SIZE*sizeof(float));
	cl::Buffer buffer_Y(context,CL_MEM_READ_ONLY, SIZE*sizeof(float));
	cl::Buffer buffer_temp(context,CL_MEM_READ_ONLY, SIZE/4*sizeof(float));
	cl::Buffer buffer_Theta(context,CL_MEM_READ_WRITE, 2*sizeof(float));//Depends on the dimension of x;
	
	/////////////Creating Queue////////////////////////
	cl::CommandQueue queue(context, default_device);
	vector<cl::Event> waitList;
	
	/////////////Writing Input Buffers////////////////////////
	cl::Event e1;
	queue.enqueueWriteBuffer(buffer_X,CL_TRUE | CL_QUEUE_PROFILING_ENABLE,0,SIZE*sizeof(float),x,NULL,&e1);
	waitList.push_back(e1);
	cl::Event e2;
	queue.enqueueWriteBuffer(buffer_Y,CL_TRUE | CL_QUEUE_PROFILING_ENABLE,0,SIZE*sizeof(float),y,NULL,&e2);
	waitList.push_back(e2);
	cl::Event e3;
	queue.enqueueWriteBuffer(buffer_temp,CL_TRUE | CL_QUEUE_PROFILING_ENABLE,0,SIZE/4*sizeof(float),temp,NULL,&e3);
	waitList.push_back(e3);
	
	//queue.enqueueBarrierWithWaitList();
	cl::Kernel kernel=cl::Kernel(program,"gradient_descent");
	//size_t wg_size;
	//int err=cl::Kernel::getWorkGroupInfo(kernel,default_device,CL_KERNEL_WORK_GROUP_SIZE,sizeof(wg_size), &wg_size, NULL);
	
	
	kernel.setArg(0, buffer_X);
	kernel.setArg(1, buffer_Y);
	kernel.setArg(2, buffer_temp);
	kernel.setArg(3, buffer_Theta);
	cout<<"Data Size:"<<SIZE<<endl;
	/////////////Starting Kernel////////////////////////
	vector<cl::Event> waitList2;
	cl::Event kerCom;
	queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(SIZE/4),cl::NullRange,&waitList,&kerCom);
	waitList2.push_back(kerCom);
	float Theta[2];
	/////////////Reading Output Buffer////////////////////////
	queue.enqueueReadBuffer(buffer_Theta, CL_TRUE,0,2*sizeof(float),Theta,&waitList2);
	queue.finish();
	
	cout<<"Theta:"<<Theta[0]<<","<<Theta[1]<<endl;

	
	/////////////Visualizing Data////////////////////////
	
	string comm="plot ";
	stringstream ss (stringstream::in | stringstream::out);
	stringstream ss2 (stringstream::in | stringstream::out);
	ss<<Theta[1];
	ss2<<Theta[0];
	string th1=ss.str();
	string th0=ss2.str();
	comm+=th1;
	comm+="*x ";
	comm+=th0;
	comm+=" , ";
	comm+="'data.txt' using 1:2\n";
	//cout<<comm;
	int Temp=comm.size();
	char Command[100];
	int a=0;
	for (a=0;a<=Temp;a++){
		Command[a]=comm[a];

	}
	Command[a]='\0';
	


	FILE *pipe=popen("gnuplot -persistent","w");
	if (pipe == NULL){
		cout<<"\nFailed";
	}
	fprintf(pipe,"set xrange[4:24]\n");
	fprintf(pipe,"set yrange[-5:27]\n");
//	fprintf(pipe,"plot 'data.txt' using 1:2\n");
	fprintf(pipe,Command);
	fclose(pipe);	


	return 0;
}
	
		
	