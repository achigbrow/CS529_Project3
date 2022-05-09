# CS529_Project3
## Project Dependencies

- FFMPEG
- Pydub
- Librosa

## Classifiers
We provide the following complete classifiers in this project:

- SVM
- Neural Network
- Ensemble Neural Network

We achieved our highest accuracy using the Ensemble Neural Network. 

There is also a Convolutional Neural Network however, it is still a work in progress.

### SVM
You can run the SVM using our pre-processed data using the following command:

```shell
python driver.py -svm
```

### Neural Network
You can run the Neural Network using our pre-processed data using the following command:

```shell
python driver.py -neuralnet
```

### Ensemble
You can run the Ensemble using our pre-processed data using the following command:

```shell
python driver.py -ensemble -ensemble_count=10
```

This will print the training and testing accuracy to `stdout`, produce a confusion 
matrix saved to a csv (`SVMconfusion.csv`), and create a submission file called 
`predictions.csv`.

If you have a directory of training `.wav` files and a directory of testing `.wav` 
files, you can run the SVM on the processed MFCCs generated from those wavs using:

```shell
python driver.py -svm_wav
```

This will print the training and testing accuracy to `stdout`, produce a confusion
matrix saved to a csv (`SVMconfusion.csv`), and create a submission file called
`predictions.csv`.

:warning: Your training directory should be called `train` and your testing 
directory should be called `test`. You need to provide the training csv as `train.csv` 
and the test csv as `test.csv` All of these items (two directories and two csvs) should 
be located in the root directory for this project.

:warning: The provided csvs must not contain a header row. Please remove it prior to running.

## File Conversions (Optional)

We have provided two CSVs (`data.csv` and `test.csv`) containing pre-calculated 
feature sets to shorten the run time for the models.

All of the visualizations require the files to be in `.wav` format. We provide a 
directory (`./visualization/wavs`) containing a sample wav for each genre type. You can find 
instructions on using these files in the [visualization](#visualization) section.

If you want to run the visualizations or the SVM function that uses wavs instead 
of the csv, you will need to convert your mp3s to wav. Do this by running the following command:

```shell
python convert_files.py -train <filepath/to/training/mp3s> \
-train_out <filepath/for/output/training/wavs> \
-test <filepath/to/test/mp3s> \
-test_out <filepath/for/output/test/wavs>
```
:warning: This process may take up to 10 minutes to complete. It requires that `ffmpeg` 
and `pydub` be installed.

## Visualization

We provide a means to recreate most of the visualizations found in our report using 
`driver.py`. These visualizations rely on wav files and so you will not get the scatterplots 
that encompassed the entire data set.

The following command will create the plots. They will open up individually. Open the next
plot by closing the one currently open. There will be a total of 13 plots.

```shell
python driver.py -visualize
```

## References

- https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
- https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
- https://pythonbasics.org/convert-mp3-to-wav/
- https://musicinformationretrieval.com/mfcc.html
- https://scikit-learn.org/stable/auto_examples/svm/plot_linearsvc_support_vectors.html
