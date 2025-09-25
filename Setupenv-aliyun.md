# Install HydraServe on Aliyun ACK

HydraServe was originally tested on [Alibaba Cloud ACK](https://www.alibabacloud.com/en/product/kubernetes?_p_lc=1). Follow the instructions below to prepare the cluster for HydraServe.

### 1. Create an Aliyun ACK Cluster

Create an ACK cluster by clicking **Create Kubernetes Cluster** at the [ACK Console](https://cs.console.aliyun.com/?#/k8s/cluster/list) page.

#### 1.1 Cluster Configurations

- Kubernetes Version: v1.30.7+
- Network Settings
  - Configure SNAT for VPC: Enable
  - Network Plug-in: Terway

#### 1.2 Node Pool Configurations

Create a node pool with zero initial nodes.
- Container Runtime: containerd
- Configure Managed Node Pool: Disable
- Volumes
  - System Disk: 200 GiB
- Instances
  - Expected Nodes: 0

#### 1.3 Component Configurations

You can disable all features.

### 2. Add ECS Instances

#### 2.1 Create ECS Instances

Create ECS instances by clicking **Create Instance** on the [ECS Console](https://ecs.console.aliyun.com/home#/) page.
- Image: Alibaba Cloud Linux
- Enable a public IP address for access
- Use the same VPC and security group as your ACK cluster
- All instances should be within the same region
- Allocate at least a 200 GB ESSD to store images

ECS types used in our latency measurement experiments (testbed (i)).

```
4 * ecs.gn7i-c32g1.8xlarge (1*A10)
4 * ecs.gn6e-c12g1.12xlarge (4*V100)
2 * ecs.gn7i-c32g1.16xlarge (32Gbps, As remote storage)
```

ECS types used in our end-to-end experiments (testbed (ii)).

```
2 * ecs.gn7i-c32g1.32xlarge (4*A10)
4 * ecs.gn6e-c12g1.12xlarge (4*V100)
6 * ecs.gn7i-c32g1.16xlarge (32Gbps, As remote storage)
```

#### 2.2 Add Instances to the ACK Cluster

1. On the Cluster Management page, navigate to Nodes -> Node Pools -> Add Existing Node to add instances to the default nodepool.
2. Select the created instances.
3. Check **Store Container and Image Data on a Data Disk**.
4. Wait for the instances to join the cluster.

#### 2.3 Configure Kubernetes Access

1. On the Cluster Management page, navigate to Cluster Information -> Connection Information -> Obtain Long-term Kubeconfig, and copy the content from the **Internal Access** section.
2. Log in to the master node (you can choose any arbitrary node that is not used for GPU inference) and run the following commands.
```
mkdir -p ~/.kube
vim ~/.kube/config
[Paste the just copied content here]
```

### 3. Enable GPU Sharing

1. On the Cluster Management page, navigate to Applications -> Cloud-native AI Suite -> Deploy.
2. Check **Scheduling Policy Extension**.
3. Click **Advanced**, and configure the `policy` field of `cgpu` to 1.
4. Deploy the suite.

### 4. Initialize the Master Node Environment

Log in to your master node and run the following commands.
```
[Clone this repo]
cd hydraserve-artifact/scripts/kubernetes
sh install_python.sh    # The Kubernetes package version must be consistent with your Kubernetes cluster version.
sh tool-node-shell/setup.sh
```
### 5. Configure Node Labels
   
If you are using ECS instances not listed in Section 2.1, please first configure the specifications of instance types in `scripts/kubernetes/vllm/src/ECSInstance.py`.

First, label all GPU servers.
```
kubectl label node [node_name] gpu_server=true --overwrite
```

Next, apply the GPU sharing labels to all relevant nodes with the following command.
```
cd hydraserve-artifact/scripts/kubernetes
SHARE=1 python label_nodes.py
```

### 6. Mount a NAS Volume to All Instances

Follow the instructions in [Mount NAS to multiple instances](https://help.aliyun.com/zh/nas/user-guide/mount-a-nas-file-system-on-multiple-ecs-instances-at-the-same-time?spm=5176.nas_overview.help.dexternal.51cf217dfzlsrW) to mount a shared NAS volume to the `/mnt` path of all instances.