# VideoAnalyticsFramework

- `dataset/`: Used to host the dataset, or point to the actual dataset
- `ground_truth/`: Used to host the ground truth results, one folder for a video
- `log/`: Used to save logs
- `plot/`: Used to save data analysis results
- `results/`: Used to host experimental results data
- `src/`:
    - `config.yaml`: configuration parameters for edge and cloud
    - `edge.py`: code for the edge side
    - `cloud.py`: code for the cloud side
    - `log.txt`: log file
    - `tools/`:
        - `video_source.py`: a loader to read in videos/images
        - `utils.py`: miscelleneous utility functions
        - `log.py`: logging scripts
        - `socket.py`: communications sockets between edge and cloud
        - `perf.py`: performance monitor
        - `eval.py`: evaluation scripts