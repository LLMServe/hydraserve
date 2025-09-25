# Setup kubectl node-shell

```
# On master node
sh setup.sh
# Test
source ~/.bashrc
kubectl node-shell [node name] -- [command]
```

# Extend cloud disks for all nodes

```
# On master node
sh ../install_python.sh
python extend_disks.py
```