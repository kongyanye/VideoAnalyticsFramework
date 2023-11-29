# VideoAnalyticsFramework


## Description

A minimal working example of a video analytics framework with object detection (YOLOv8n) as the backend task.


## Features:
- Two roles: `edge` to send frames, and `cloud` to infer
- Communication: implemented with Socket, lightweight and efficient
- Performance monitor: an easy implementation to log and show performance
- Evaluator: automatically generate ground truth results and evaluate results
- Multi-thread: a single procedure implemented as a thread, working in pipelines
- Multi-task running: easy to run a bunch of task in one script
- Plot example: an actual example of how to plot the expected figure 


## File Structure:
- `dataset/`: used to host the dataset, or point to the actual dataset
- `ground_truth/`: used to host the ground truth results, one folder for a video
- `log/`: used to save logs
- `models/`: used to host model checkpoints
- `plot/`: used to save data analysis results
- `results/`: used to host experimental results data
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
        - `image_codecs.py`: encoding and decoding images


## Requirements
```
pip install ultralytics
pip install SciencePlots
```
others are common packages


## Tips
- Modify code to fit your own scenario, e.g., task, role of edge and cloud, encoding/decoding, frame filtering, etc
- Best to check actual performance with a screen for debugging