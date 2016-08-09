__kernel void gradient_descent(__global float4* x, __global float4* y, __global float2* t, __global float2* th){

	uint global_addr=get_global_id(0);
	uint local_addr=get_local_id(0);
	float4 X=x[global_addr];
	float4 Y=y[global_addr];
	//X.s0=0.98f;
	float2 Th;
	printf("in=<%f,%f>\n",X.s0,X.s1);


	float alpha=0.01f;
	float Theta[2];
	
	for(int i=0;i<1000;i++){
	Th=th[0];
	Theta[0]=Th.s0;
	Theta[1]=Th.s1;
	float4 h_theta;
	h_theta.s0=Theta[0]+Theta[1]*X.s0;
	h_theta.s1=Theta[0]+Theta[1]*X.s1;
	h_theta.s2=Theta[0]+Theta[1]*X.s2;
	h_theta.s3=Theta[0]+Theta[1]*X.s3;

	float4 diff=h_theta - Y;
	//printf("v=<%f,%f>\n",t[local_addr].s0,t[local_addr].s1);
	
	float4 temp[2];
	temp[0]=diff*1;
	temp[1]=diff*X;
	//printf("temp=<%f,%f,%f>\n",temp[1].s0,diff.s0,X.s0);
	t[local_addr].s0=temp[0].s0+temp[0].s1+temp[0].s2+temp[0].s3;
	t[local_addr].s1=temp[1].s0+temp[1].s1+temp[1].s2+temp[1].s3;
	//printf("out=<%f,%f>\n",t[local_addr].s0,t[local_addr].s1);

	barrier(CLK_LOCAL_MEM_FENCE);
	float2 sum;



		if(get_local_id(0) == 0){
			sum.s0=0;
			sum.s1=0;
			for(int i=0;i<get_local_size(0);i++){
				sum=sum+t[i];
			}
			Theta[0]=Theta[0]-(alpha/96)*(sum.s0);
			Theta[1]=Theta[1]-(alpha/96)*(sum.s1);
			sum.s0=Theta[0];
			sum.s1=Theta[1];
			th[get_group_id(0)]=sum;


		}

	}


}


