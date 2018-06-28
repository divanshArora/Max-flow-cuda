/*
This works for all.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <float.h>
#include <math.h>
#include <iostream>
#include <climits>
#include <vector>
#include <stack>
using namespace std;
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>



struct edge{ int flow,capacity,to; edge(int to,int capacity){ this->to=to; this->capacity=capacity; this->flow=0; } };
struct node{ int height,excess,color; vector<edge> edges; };

node* graph;

//functions
void push_relabel_GPU(int N, node * graph);



void addEdge(int from,int to,int capacity){
	graph[from].edges.push_back(edge(to,capacity));
}

void findMinCut(int N)
{
	int src=0,sink=N-1; // src should be background

	// compute preflow
	for(int i=0;i<N;i++){ graph[i].height=0; graph[i].excess=0; } graph[src].height=N;
	for(vector<edge>::iterator e=graph[src].edges.begin();e!=graph[src].edges.end();e++){
		graph[e->to].excess=e->capacity; e->flow=e->capacity;
		addEdge(e->to,src,e->flow);
	}

	bool hasExcessNode=true;
	while(hasExcessNode)
	{
		hasExcessNode=false;
		for(int i=1;i<N-1;i++) if(graph[i].excess>0)
		{
			// push from i to neighbours
			hasExcessNode=true; bool pushed=false;
			for(vector<edge>::iterator e=graph[i].edges.begin();e!=graph[i].edges.end();e++) if(graph[e->to].height<graph[i].height and e->capacity>e->flow)
			{
				int del=min(graph[i].excess,e->capacity-e->flow);
				e->flow+=del; graph[i].excess-=del; graph[e->to].excess+=del;

				// update residual graph
				bool edgeFound=false;
				for(vector<edge>::iterator e2=graph[e->to].edges.begin();e2!=graph[e->to].edges.end();e2++)
					if(e2->to==i){ e2->flow-=del; edgeFound=true; break; }
				if(!edgeFound) addEdge(e->to,i,del);

				pushed=true; break;
			}

			if(!pushed)
			{
				// relabel i to enable push afterwards
				int minHeight = INT_MAX;
				for(int j=0; j<graph[i].edges.size();j++)
					{
						edge e = graph[i].edges[j];
						if(e.to!=sink and e.capacity>e.flow){
							minHeight=min(minHeight,graph[e.to].height);
						}
					}
				if(graph[i].height<=minHeight) graph[i].height=minHeight+1;
			}
		}
	}

	// do a dfs from src to mark background pixels
	stack<int> stack; stack.push(src);
	while(!stack.empty())
	{
		int curr=stack.top(); stack.pop();
		graph[curr].color=0; // mark the pixel as background

		for(vector<edge>::iterator e=graph[curr].edges.begin();e!=graph[curr].edges.end();e++)
			if(e->capacity==e->flow and graph[e->to].color!=0) stack.push(e->to);
	}
}
// 0 , 1 , 2 , N = 3
// (0,1,2),(3,4),(5,6,7) assume num_neighbours = 2
//[2][0] gives 3 + 1*2 + 0 = 5
//[1][1] gives 3 + 0*2 + 1 = 4
//[0][2] gives
__host__ __device__ int get_index(int i, int j, int num_neighbours, int N){
//	cout<<"got i = "<<i<<" j = "<<j<<" N = "<<N<<'\n';
	int ans = 0;
	if(i>0)
	{
		ans+=N;
		ans+=(i-1)*num_neighbours;
		ans+=j;
	}
	else
	{
		ans+=j;
	}
//	cout<<"returning ans = "<<ans<<'\n';
	return ans;
}

int main()
{
	int N=6;
	graph=new node[N];
	for(int i=0;i<N;i++) graph[i].color=1;

	/*addEdge(0,1,3);
	addEdge(0,2,2);
	addEdge(1,2,5);
	addEdge(2,3,3);
	addEdge(1,3,2);*/

    addEdge(0, 2, 13);
	addEdge(0, 1, 16);
    addEdge(1, 2, 10);
//    addEdge(2, 1, 4);
    addEdge(1, 3, 12);
    addEdge(2, 4, 14);
    addEdge(3, 2, 9);
    addEdge(2, 3, 9);
    addEdge(3, 5, 20);
    addEdge(4, 3, 7);
    addEdge(4, 5, 4);

//    findMinCut(N);
//    for(int i=0;i<N;i++) if(graph[i].color==0) cout<<i<<" "; cout<<endl;
 //   cout<<graph[N-1].excess;
    push_relabel_GPU(N,graph); //Make sure findMinCut is NOT called before this as it modifies graph.
}

__global__ void kernel(int * height_d, int * excess_d,int * adjacency_list_d,int * size_matrix_d, int * capacity_d, int * flow_d, int N, int num_neighbours, int sink){
	int cycle = 1;
	int u = blockIdx.x*blockDim.x+threadIdx.x;
	while(cycle>0)
	{
		printf("Working on node/thread %d\n",u);
		if(excess_d[u]>0 && u!=sink)
		{
			int e_dash = excess_d[u];
			int h_dash  = 100000;
			int v_dash = -1;
			int i_dash = -1;

			for(int i=0;i<size_matrix_d[u];i++)
			{
				int ind = get_index(u,i,num_neighbours,N);
				int v = adjacency_list_d[ind];
				int h_da_da = height_d[v];
				if(h_da_da<h_dash && ((capacity_d[ind] - flow_d[ind])>0))
				{
					v_dash = v;
					i_dash = i;
					h_dash = h_da_da;
				}
			}
			if(height_d[u]>h_dash)
			{
				printf("nearest neighbour with lower height %d\n",v_dash);
				int d = 0;
				int x_tmp=capacity_d[get_index(u,i_dash,num_neighbours,N)]-flow_d[get_index(u,i_dash,num_neighbours,N)];
				if(x_tmp<0)
				{
					//assert(false)
				}
				if(e_dash<x_tmp)
				{
					d = e_dash;
				}
				else{
					d = x_tmp;
				}
				int ind_of_u_in_v_list = 0;
				for(int i=0;i<size_matrix_d[v_dash];i++)
				{
					if(adjacency_list_d[get_index(v_dash,i,num_neighbours,N)]==u)
					{
						ind_of_u_in_v_list = i;
						break;
					}
				}
				atomicAdd(&flow_d[get_index(u,i_dash,num_neighbours,N)],d);
//				atomicSub(&flow_d[get_index(v_dash,ind_of_u_in_v_list,num_neighbours,N)],d);
				atomicSub(&excess_d[u],d);
				atomicAdd(&excess_d[v_dash],d);
			}
			else{
				height_d[u]= h_dash +1;
			}
		}
		cycle-=1;
	}
}


void global_relabel(int * height, int * excess,int * adjacency_list,int * size_matrix, int * capacity, int * flow, int N, int num_neighbours, int src, int sink)
{
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<size_matrix[i];j++)
		{
			int ind = get_index(i,j,num_neighbours,N);
			int u = i, v = adjacency_list[ind];
			int ind_of_v_in_u_list =-1;
			if(height[u]>height[v]+1)
			{
				int cfuv = (capacity[get_index(u,j,num_neighbours,N)]-flow[get_index(u,j,num_neighbours,N)]);
				excess[u] =excess[u] - cfuv;
				excess[v] =excess[v] + cfuv;
				for(int k=0;k<size_matrix[v];k++)
				{
					if(adjacency_list[get_index(v,k,num_neighbours,N)]==u)
					{
						ind_of_v_in_u_list = k;
						break;
					}
				}
				int ind1 = get_index(v,ind_of_v_in_u_list ,num_neighbours,N);
//				int cfvu = capacity[ind1] - flow[ind1];
				flow[ind1] =  flow[ind1] - cfuv;
				flow[ind] = capacity[ind];


			}
		}
	}

}

void print_flow(int * flow,int N, int * size_matrix, int * adjlist, string s, int * excess)
{
	if(s=="excess")
	{

		for(int i=0;i<N;i++)
		{
			cout<<"Excess for "<<i<<' '<<excess[i];
		}cout<<'\n';
		return;
	}
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<size_matrix[i];j++)
		{
			int ind = get_index(i,j,16,N);
			cout<<s+" for "<<i<<" and "<<adjlist[ind]<<" = ";

			cout<<" "<<flow[ind]<<" ";
		}
		cout<<'\n';

	}
}



void push_relabel_GPU(int N, node * graph)
{
	int src=0,sink=N-1; // src should be background
	// compute preflow
//	int * height_arr = malloc(sizeof(int))

	size_t nsize = sizeof(int)*N;
	int NUM_NEIGHBOURS = 16;
	size_t twonsize = sizeof(int)*((N-2)*NUM_NEIGHBOURS + 2*N);

	//CPU variables
	int * height = (int*)malloc(nsize);
	int * excess = (int*)malloc(nsize);
	int * adjacency_list = (int*)malloc(twonsize);
	int * size_matrix = (int*)malloc(nsize);
	int * capacity = (int*)malloc(twonsize);
	int * flow = (int*)malloc(twonsize);
	int * cf = (int*)malloc(twonsize);
	//GPU variables

	int * height_d; cudaMalloc(&height_d, nsize);
	int * excess_d; cudaMalloc(&excess_d, nsize);
	int * adjacency_list_d; cudaMalloc(&adjacency_list_d,twonsize);
	int * size_matrix_d; cudaMalloc(&size_matrix_d, nsize);
	int * capacity_d; cudaMalloc(&capacity_d,twonsize);
	int * flow_d; cudaMalloc(&flow_d,twonsize);
	int * cf_d = (int*)malloc(twonsize);



	//Setting values for new AoS implementation
	memset(height,0,nsize);
	memset(excess,0,nsize);
	for(int i=0;i<N;i++)
	{
		size_matrix[i]=0;
	}

	for(int i=0;i<N;i++)
	{
		int s = graph[i].edges.size();
//		size_matrix[i] = s;
		for(int j=0;j<s;j++)
		{
			cout<<" -------------- << s = << "<<s<<'\n';
//			int u = get_index(i,j,NUM_NEIGHBOURS,N);
//			cout<<"i = "<<i<<" j = "<<j<< "setting "<< u<<"adj_list[ ] as "<< (graph[i].edges)[j].to<<'\n';
			int v = (graph[i].edges)[j].to;
			int cap = (((graph[i].edges)[j]).capacity);
			adjacency_list[get_index(i,size_matrix[i],NUM_NEIGHBOURS,N)] = v;
			capacity[get_index(i,size_matrix[i],NUM_NEIGHBOURS,N)] = cap;
			flow[get_index(i,size_matrix[i],NUM_NEIGHBOURS,N)] = 0;
			cf[get_index(i,size_matrix[i],NUM_NEIGHBOURS,N)] = capacity[get_index(i,size_matrix[i],NUM_NEIGHBOURS,N)]-flow[get_index(i,size_matrix[i],NUM_NEIGHBOURS,N)];
			size_matrix[i]++;

			adjacency_list[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)] = i;
			capacity[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)] = cap;
//			capacity[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)] = 0;
			flow[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)] = 0;
			cf[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)] = capacity[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)]-flow[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)];
			size_matrix[v]++;


//			capacity[get_index(v,size_matrix[v],NUM_NEIGHBOURS,N)] = i;

//			capacity[get_index(i,j,NUM_NEIGHBOURS,N)] = (((graph[i].edges)[j]).capacity);


			//cout<<"i = "<<i<<" j = "<<j<< "setting capacity "<< u<<"adj_list[ ] as "<< (graph[i].edges)[j].capacity<<'\n';
		}
	}

	//excess is the excess flow of vertex, flow is the flow on edge
	//Initializing pre-flow
	for(int i=0;i<N;i++){ height[i]=0; excess[i]=0; } height[src]=N;

//	for(int i = 0;i<size_matrix[src];i++){
//		//v is the to vertex
//		int v = adjacency_list[get_index(src,i,NUM_NEIGHBOURS,N)];
//		excess[v]=capacity[get_index(src,i,NUM_NEIGHBOURS,N)];
//		int cap = capacity[get_index(src,i,NUM_NEIGHBOURS,N)];
//		flow[get_index(src,i,NUM_NEIGHBOURS,N)]= cap;
//		cout<<"set excess of "<<v<<" to "<<excess[v];
//		//add-edge implementation
//		//!!!!! CHECK 7
//		int last_elem = size_matrix[v];
//		cout<<"last elem of "<<v<<" = "<<last_elem<<'\n';
////		adjacency_list[get_index(v,last_elem,NUM_NEIGHBOURS,N)] = src;
////		cout<<" setting edge in list "<<get_index(v,last_elem,NUM_NEIGHBOURS,N)<<'\n';
////		capacity[get_index(v,last_elem,NUM_NEIGHBOURS,N)] = 0;
////		flow[get_index(v,last_elem,NUM_NEIGHBOURS,N)] = -flow[get_index(src,i,NUM_NEIGHBOURS,N)];
////		cout<<"flow = "<<flow[get_index(v,last_elem,NUM_NEIGHBOURS,N)]<<'\n';
////		size_matrix[v]++;
////		addEdge(e->to,src,e->flow);
//		//add edge fn ends or include next line too
//		excess[src] -= flow[get_index(src,i,NUM_NEIGHBOURS,N)];
//		cout<<"Loop end---------------------------------"<<'\n';
//	}

	for(int i=0;i<size_matrix[src];i++)
	{
		int index_v = get_index(src, i, NUM_NEIGHBOURS, N);
		int cap = capacity[index_v];
		int v = adjacency_list[index_v];
		flow[index_v] = cap;
		excess[src] -= cap;
		excess[v]  = cap;
		//uncomment loop cfor correct termination
//		for(int j=0;j<size_matrix[v];j++)
//		{
//			int ind = get_index(v,j,NUM_NEIGHBOURS,N);
//			if(adjacency_list[ind]==src)
//			{
//				flow[ind] = -cap;
//				break;
//			}
//		}
	}


	height[src] = N;
	//pre-flow ends

	//Copying

	cudaMemcpy( excess_d, excess, nsize, cudaMemcpyHostToDevice);
	cudaMemcpy(capacity_d, capacity,twonsize, cudaMemcpyHostToDevice);
	cudaMemcpy(flow_d, flow,twonsize, cudaMemcpyHostToDevice);
	cudaMemcpy(size_matrix_d, size_matrix,nsize, cudaMemcpyHostToDevice);
	cudaMemcpy(adjacency_list_d, adjacency_list,twonsize, cudaMemcpyHostToDevice);

	//Starting main loop
	cout<<"graph[src].excess = "<<excess[src]<<" graph[sink].excess = "<<excess[sink]<<'\n';
	int cnt =1000;
	while(excess[src]+ excess[sink]< 0 && cnt>0)
	{
		cout<<"graph[src].excess = "<<excess[src]<<" graph[sink].excess = "<<excess[sink]<<'\n';
		cudaMemcpy(height_d, height, nsize, cudaMemcpyHostToDevice);
		//call kernel here
		kernel<<<1,N>>>(height_d,excess_d,adjacency_list_d,size_matrix_d,capacity_d,flow_d,N,NUM_NEIGHBOURS,sink);

		cudaMemcpy(height, height_d, nsize, cudaMemcpyDeviceToHost);
		cudaMemcpy( excess, excess_d, nsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(capacity, capacity_d,twonsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(flow, flow_d,twonsize, cudaMemcpyDeviceToHost);
		cudaMemcpy(size_matrix, size_matrix_d,nsize, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cout<<"H array: --------------------------------------------\n";
		for(int q=0;q<N;q++)
		{
			cout<<height[q]<<' ';
		}
		printf("FLOW:\n");
		print_flow(flow,N,size_matrix, adjacency_list,"flow",excess);

		printf("EXCESS:\n");
		print_flow(flow,N,size_matrix, adjacency_list,"excess",excess);

		printf("capacity:\n");
		print_flow(capacity,N,size_matrix, adjacency_list,"capacity",excess);

		cnt--;
		//global_relabel(height,excess,adjacency_list,size_matrix,capacity,flow,N,NUM_NEIGHBOURS,src,sink);
		//call global relabel here
	}

}

