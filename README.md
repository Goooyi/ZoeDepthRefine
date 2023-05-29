# ZoeDepthRefine

A small network to improve ZoeDepth on HabitatDyn Dataset

Related-reading for better understand the problem:

[Paper:Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/pdf/1406.2283.pdf)

[Paper:On regression losses for deep depth estimation](https://hal.science/hal-01925321/document)

[Paper:On the Metrics for Evaluating Monocular Depth Estimation](https://arxiv.org/pdf/2302.10007.pdf)

[Paper: ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288)

[Blog:Monocular Depth Estimation using ZoeDepth : Our Experience](https://medium.com/@bhaskarbose1998/monocular-depth-estimation-using-zoedepth-our-experience-42fa5974cb59)

TODO:
- [ ] standalone config file, 
- [ ] how to pass parameter to train() cleaner? wraper function? the example from [pytorch-ratytune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#hyperparameter-tuning-with-ray-tune) use load_data() and pass a `data_dir` to trainer
- [ ] better val video clip to cover most range in HabitatDyn dataset
- [ ] ray tun: Use metric to determin which loss-fun, epoch, lr, is best combo
- [ ] Better Loss function
- [ ] val and test
- [ ] implement Metrics used to compare different model: Relative-Square, Relative-absolute and RMS
- [ ] extend this repo as a standalone method that does Relative Depth Estimation (RDE) to Metric Depth Estimation (MDE), since this should be a small plug-in module as the RDE model doing the hard work and this module only trailer to specfic dataset with metric depth info. Application?

### Metrcis to evaluate different loss function.

MAE SquaRel RMSE RMSLE Delta

# HYPERPARAMETER TUNING WITH RAY TUNE

Original [Blog](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html#hyperparameter-tuning-with-ray-tune)


# Research: Dataloader bottele neck: The PyTorch IO problem

Local Specs: 
```
Pytroch model: 
    pytorch Unet model with ResNet18 backbone
Hardware spec: 
    CPU: AMD Ryzen 9 3900X 12-Core Processor
    GPU: RTX3090
    Disk: Unknown
Data:
    image data of size 480*640 rgb.jpg and depth.png
    resizing to 192*256 and concatenate in Dataset class using torch
    1 video clip of 240 frames rgb.jpg is 15.5MB on hard disk
    1 depth folder with corresponding rgb.png file 3.5MB 
    1 pseudo depth of 16bit format from zoe model is 16 MB
    so for 1 video about 35MB is loaded to memory
    1 scene has about 54 videos, 10 scene will have about 20GB memory usage: but 27 
```

For batch size 64, shuffle=True, pin_memory=True, num_workers=4,prefetch_factor=2: 

```logging
dataloader time: 2.9199090003967285 
memory to cuda time: 0.03172612190246582 
train train 1 batch time:  0.08431768417358398 
```
Observe lsage around 35% for all thread, GPU usage peak to 99% but only for a short period cause the datalloding bottleneck


## Solutions

- [x] not doing transform will save 2 second for each folder, so doing transfrom using cuda and pin_memory=False
- [ ]  Usually, big gains over torch.DataParallel using apex.DistributedDataParallel. Moving from ‘one main process + worker process + multiple-GPU with DataParallel’ to 'one process-per GPU with apex (and presumably torch)
DistributedDataParallel has always improved performance for me. Remember to (down)scale your worker processes per training process accordingly. Higher GPU utilization and less waiting for synchronization usually results, the variance in batch times will reduce with the average time moving closer to the peak.
- [ ] multi-treading vs Distributed(multi-server) vs Data Parallel(multi-gpu) vs multi-gpu(https://pytorch.org/docs/stable/notes/multiprocessing.htm)
- [ ] voom voom repo
```python
for epoch in tqdm(range(1000),desc="EPOCH", position=0):
    if epoch%4 == 0:
        print(f'change the scenes\n')
        datasets = []
        for record_file in tqdm(random.sample(dataset_list,432), desc= "Load the datasets"):
            record = np.load(dataset_path + record_file,allow_pickle=True).tolist()
            datasets += (record)
        print(f'change the scenes finished data length: {len(datasets)}\n')
        imagination_dataset = ImaginationDataset(datasets=datasets, config=config)
        train_dataloader = DataLoader(imagination_dataset, batch_size=batch_size, shuffle=True)
```
- [ ] Check amp implementation old vs new 
- [ ] Nvidia DALI
- [ ] [Best Practing](https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf)
- [ ] setting to gradient to none instead of zero grad for performance small improvements.
- [ ] when set `shuffle=True` and `sampler is None` in Dataloader, the RandomSampler is shuffle without replacement as default which make the model harder to train as the model receive total different data in each batch, but commonly make the final model perform better.
- [x] set `x.cuda(non_blocking=True)` [if pin_memory = True](https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader)
- [x] Preprocess input data in a fast database format like hdf5 (h5py) ,lmdb or [webdataset](https://github.com/webdataset/webdataset), or [custom streaming data loader class](https://jamesmccaffrey.wordpress.com/2021/03/08/working-with-huge-training-data-files-for-pytorch/)
- [x] Load data into memory in `Dataset:__init__`, then if the dataset is too big to load to memeory, write a streaming data loader class or using mini dataset during epoch iter like Dr.Shen's proposal below. But if the disk read IO speed is the bottleneck, this only helps to reduce data transfer overhead.
- [x] do batched data augmentations o the GPU using Kornia.
- [x] num_workers too high could be problematic(https://pytorch.org/docs/stable/data.html#multi-process-data-loading)
- [x]  Use torchData instead of dataloader might help, not helping if the disk IO is the bottelneck and torchData also do not support batch transformation either.
- [x] Dataset return a CUDA tensor is [problematic when using pin_memory=True](https://pytorch.org/docs/stable/data.html#multi-process-data-loading)
- [x] Dataset using PIL to read image is [problematic](https://discuss.pytorch.org/t/how-to-prefetch-data-when-processing-with-gpu/548/19), use pillow-sims helps about 20% improvement
- [x] When there is heavy data processing in Dataset class, set num_workers>1 can help but when IO is bottleneck, all worker thread waiting for IO and thus making it even worse.
- [x] avoid using python object in Dataset, instead use victorized operations like numpy, torch. 
- [x] setting cu-DNN auto-benchmark to true
- [x] Use LRU Cache if same data is used in batches because of data augmentation, etc.
- [x] Use Image.open(...).cpoy(): otherwise the server willdown cause it will not close the Image object, .copy() will dereference ImageFile object, call destructor 
- [x] random resized crop to the dimension of 224x224