# Transformer pipeline:

### For training edit the configs for file paths for datasets and run the main_transformers.py
### For Evaluation: Use: python -m TRANSFORMER.cli file --model-path <path to saved model>
### Note that you should use that command from the Root dir.
# The evaluation outputs .csv files for each dataset which I used when doing analysis for the thesis.


### Python version: 3.11 is supported.


### IMPORTANT: For incompatibility issues with ffmpeg (bit if you get it working use newer versions) we use older numpy and datasets versions. THis works if you edit: \datasets\formatting\formatting.py so that return arrays in lines: 196-197 have arguments like this:

return np.array(array, dtype=object)
        return np.array(array)