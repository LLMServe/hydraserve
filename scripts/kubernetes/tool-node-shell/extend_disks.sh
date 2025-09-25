set -eu
# 1. 安装growpart工具【您当前的分区类型为GPT，还需安装gdisk工具】 
type growpart || yum install -y cloud-utils-growpart
type sgdisk || yum install -y gdisk
# 
# 2. 扩容分区：运行命令 growpart /dev/vda 3 
LC_ALL=en_US.UTF-8 growpart /dev/vda 3
# 扩容文件系统：您当前的文件系统为ext4，因此运行resize2fs命令 
resize2fs /dev/vda3