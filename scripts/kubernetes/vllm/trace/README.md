# Trace Replay

We replay trace based on the implementation in [AlpaServe](https://github.com/alpa-projects/mms/blob/main/alpa_serve/trace/README.md).


## Dataset
This folder provides methods to generate a TraceReplay from a public trace. Supported public trace:
- Microsoft azure_v2 trace. [[Introduction]](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md) [[Download]](https://drive.google.com/file/d/1IOVoUoodBj4aKeyggxMnEVChEPutN4t7/view?usp=sharing)


## How to use

```
pip install modelscope==1.13.3 matplotlib
python generate_trace.py [PATH_TO_AZURE_TRACE]
```
- A ```trace.pkl``` will be generated, which contains a one-day trace.