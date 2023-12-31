# VideoAnalyticsFramework


## Description

A minimal working example of a video analytics framework with object detection (YOLOv8n) as the backend task.


## Features:
- Self-contained: all you need is contained in this repo, e.g., the model checkpoint, the dataset, the example log file and results plot, etc
- Two roles: `edge` to send frames, and `cloud` to infer
- Communication: implemented with Socket, lightweight and efficient
- Performance monitor: an easy implementation to log and show performance
- Evaluator: automatically generate ground truth results and evaluate results
- Multi-thread: a single procedure implemented as a thread, working in pipelines
- Multi-task running: easy to run a bunch of task in one script
- Plot example: an actual example of how to plot the expected figure 

## Usage:
1. Modify `HOME_DIR` in `src/config.yaml` to the actual location, change `show_image` to False if you do not have a screen available
2. In one terminal, `cd src/ && python edge.py`
3. In another terminal, `cd src/ && python cloud.py`
4. Enjoy!

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
- Best to set all path in the code to absolute path starting with `/`, because the code will be called by other scripts (i.e., `run.py`)
- If you have a bug in the code when modifying it, the `edge.py` and `cloud.py` may not exit correctly, which will result in the `socket` to be still in use without the resource being released. Just wait one or two minutes and then it would be good again.
