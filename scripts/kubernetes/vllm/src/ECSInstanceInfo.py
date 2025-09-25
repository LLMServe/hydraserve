'''
ECS instance type in format:
1. GPU server: (gpu number, gpu mem in GB, network bandwidth in Gbps, memory, vCPU)
2. Storage server: bandwidth in Gbps
'''
AliyunECSInstanceInfo = {
    # T4 instances
    "ecs.gn6i-c4g1.xlarge": (1, 14, 4, 15, 4),
    "ecs.gn6i-c8g1.2xlarge": (1, 14, 5, 31, 8),
    "ecs.gn6i-c16g1.4xlarge": (1, 14, 6, 62, 16),
    "ecs.gn6i-c24g1.6xlarge": (1, 14, 7.5, 93, 24),
    "ecs.gn6i-c40g1.10xlarge": (1, 14, 10, 155, 40),
    "ecs.gn6i-c24g1.12xlarge": (2, 14, 15, 186, 48),
    "ecs.gn6i-c24g1.24xlarge":  (4, 14, 30, 372, 96),
    # A10 instances
    "ecs.gn7i-c8g1.2xlarge": (1, 22, 16, 30, 8),
    "ecs.gn7i-c16g1.4xlarge": (1, 22, 16, 60, 16),
    "ecs.gn7i-c32g1.8xlarge": (1, 22, 16, 188, 32),
    # "ecs.gn7i-c32g1.16xlarge": (2, 22, 32, 376, 64),
    "ecs.gn7i-c32g1.32xlarge": (4, 22, 64, 752, 128),
    "ecs.gn7i-c48g1.12xlarge": (1, 22, 16, 310, 48),
    "ecs.gn7i-c56g1.14xlarge": (1, 22, 16, 346, 56),
    "ecs.gn7i-2x.8xlarge": (2, 22, 16, 128, 32),
    "ecs.gn7i-4x.8xlarge": (4, 22, 16, 128, 32),
    "ecs.gn7i-4x.16xlarge": (4, 22, 32, 256, 64),
    "ecs.gn7i-8x.32xlarge": (8, 22, 64, 512, 128),
    "ecs.gn7i-8x.16xlarge": (8, 22, 32, 256, 64),
    # V100 instances
    "ecs.gn6e-c12g1.3xlarge": (1, 32, 5, 92, 12),
    "ecs.gn6e-c12g1.12xlarge": (4, 32, 16, 368, 48),
    "ecs.gn6e-c12g1.24xlarge": (8, 32, 32, 736, 96),
    # Remote Storage Network Bandwidth (Gbps)
    "ecs.gn7i-c32g1.16xlarge": 32,
    "ecs.c7.32xlarge": 48,
    "ecs.c7a.32xlarge": 32,
    "ecs.c7a.16xlarge": 16,
}